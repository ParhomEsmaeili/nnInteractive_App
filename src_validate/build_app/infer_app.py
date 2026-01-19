#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import copy 
import warnings
app_local_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
ckpt_dir = os.path.join(app_local_path, 'ckpt', 'nnInteractive_v1.0')
sys.path.append(app_local_path)
# from pathlib import Path
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnunetv2.utilities.helpers import empty_cache


class InferApp:
    def __init__(self, infer_device, algorithm_state, enable_adaptation, algo_cache_name):
        self.infer_device = infer_device
        if self.infer_device.type != 'cuda':
            raise ValueError('This script only should be used with CUDA inference device.')

        self.autoseg_infer = False #This is a variable for storing the action taken in the instance where there is no prompting information provided in a slice.
        #In the case where it is True, a prediction will be made, and the stored pred and output pred will be the same.
        #In the case where it is False, a prediction will not be made, the stored pred will be None, and the output pred will be zeroes.
        self.permitted_prompts = ('points', 'bboxes', 'scribbles', 'lasso') 
        self.prompt_subtypes = {
            'points':'free_prompts',
            'scribbles':'free_prompts', 
            'bboxes': 'partition_prompts',
            'lasso':'partition_prompts'
        }
        self.app_params = {
            'autoseg_infer_bool': self.autoseg_infer,
            'permitted_prompts': self.permitted_prompts,
            'prompt_subtypes': self.prompt_subtypes
        }
        self.load() 
        self.build_inference_apps() 

    def load(self):
        session = nnInteractiveInferenceSession(
            device=torch.device('cuda', 0),
            use_torch_compile=False,
            verbose=False,
            torch_n_threads=os.cpu_count(),
            do_autozoom=True,
            use_pinned_memory=True
        )
     
        app_params = session.initialize_from_trained_model_folder(
            model_training_output_dir=ckpt_dir,
            use_fold=0,
            checkpoint_name='checkpoint_final.pth'
        )
        self.session = session #We are going to heavily lean on the existing session implementation, and just wrap
        #it to fit the expected API

        self.app_params.update(app_params)
    
    def app_configs(self):
        #STRONGLY Recommended: A method which returns any configuration specific information for printing to the logfile. Expects a dictionary format.
        return self.app_params 



    def build_inference_apps(self):
        #Building the inference app, needs to have an end to end system in place for each "model" type which can be passed by the request: 
        # 
        # IS_autoseg, IS_interactive_init, IS_interactive_edit. (all are intuitive wrt what they represent.) 
        
        self.infer_apps = {
            'IS_autoseg':{'binary_predict':self.binary_inference},
            'IS_interactive_init': {'binary_predict':self.binary_inference},
            'IS_interactive_edit': {'binary_predict':self.binary_inference}
            }

    def binary_inference(
        self,
        request: dict,
        ) -> torch.Tensor:
        """
        Stub performing **one** forward pass of your model.

        
        bbox : list of dict | None
            Bounding‑box prompt(s).  The dict structure is shown in the challenge
            description; may be absent in refinement iterations.
        clicks : list of dict | None
            Fore‑ and background click dictionaries for every class.
        prev_pred : (D, H, W) np.ndarray | None
            Segmentation from the previous iteration.  May be `None` for the first
            call.

        Returns
        -------

        seg : np.ndarray, dtype=uint8
            Multi‑class segmentation mask.  Background **must** be 0;
            classes start from 1 … N.  Make sure dtype is `np.uint8`.
        """

        init, affine, is_state = self.binary_subject_prep(request)
        self.binary_place_interactions(init, is_state)
        pred, probs_tensor = self.binary_predict() 
        return pred, probs_tensor, affine

    def binary_place_interactions(self, init: bool, is_state: dict):
        #NOTE: nnInteractive performs best when interactions are placed in a sequential order, i.e., each prompt instance is added one after the other.
        # This is because of 2 reasons: 1) the interaction memory is configured to downregulate older interactions, 2) the zoom levels/center are set
        # based on the last interaction placed. Hence, it is expected that the performance will be suboptimal if multiple prompt instances
        # are provided at once. 

        #Given that the foreground points should always be in the vicinity of the target (whereas a background prompt may not be), we will place 
        # the foreground prompt last, so that this is the one that determines the zoom level/center.

        if not bool(is_state):
            raise Exception('Cannot be an interactive request without interaction state! Should not have reached this point!')

        #Extracting the prompt dictionaries from the interaction state.
        p_dict = (is_state['interaction_torch_format']['interactions'], is_state['interaction_torch_format']['interactions_labels'])
        #Determine the prompt types from the input prompt dictionaries
        provided_ptypes = list(set([k for k,v in p_dict[0].items() if v is not None]) & set([k[:-7] for k,v in p_dict[1].items() if v is not None]))
        #Lets provide somewhat of a reasonable limitation, which is that more than one prompt type cannot be provided at once.
        provided_subtypes = set([self.prompt_subtypes[ptype] for ptype in provided_ptypes])

        if not len(provided_ptypes) == 1:
            raise Exception('More than one prompt type was provided in the interactive request, cannot proceed with interactive inference!')
        if not len(provided_subtypes) == 1:
            raise Exception('More than 1 prompt subtype was provided, cannot proceed with interactive inference!')
        #Somewhat redundant check, but we will keep it here for now.

        if provided_ptypes[0].title() == "Points":
            points = p_dict[0][provided_ptypes[0]]
            points_lbs = p_dict[1][provided_ptypes[0] + '_labels']

            #Placing the background points first, then the foreground points.
            bg_code = self.configs_labels_dict['background']
            fg_code = self.configs_labels_dict[[k for k in self.configs_labels_dict.keys() if k != 'background'][0]]

            if bg_code != 0:
                raise Exception('Script written assuming background is assigned class 0! Cannot proceed with inference!')

            #First we add the background points, then the foreground. We always want the center of the last interaction to be the foreground class,
            #as this is most likely to be within the region of interest of the target.
            if bg_code in points_lbs:
                bg_idx = (torch.cat(points_lbs) == bg_code).nonzero(as_tuple=True)
                for idx in bg_idx[0]:
                    self.session.add_point_interaction(
                        tuple(points[idx].flatten().tolist()),
                        include_interaction=False,
                        run_prediction=False
                    )
            if fg_code in points_lbs:
                fg_idx = (torch.cat(points_lbs) == fg_code).nonzero(as_tuple=True)
                for idx in fg_idx[0]:
                    self.session.add_point_interaction(
                        tuple(points[idx].flatten().tolist()),
                        include_interaction=True,
                        run_prediction=False
                    )
                
        elif provided_ptypes[0].title() == "Scribbles":
            raise NotImplementedError('Conversion from api-structure to array form not yet implemented')
            
        elif provided_ptypes[0].title() == "Bboxes":   
            #The pre-trained model is trained with 2D bounding boxes in mind. The API will use a convention 
            # that any of the coordinates must be matching. E.g. x_min = x_max, etc, to indicate this. However, the format expected for a 2D bounding box
            # in nnInteractive is to have this represented as the coordinate having difference 1. This is presumably because the bbox is represented as 
            # an array in the network input.

            bboxes = p_dict[0][provided_ptypes[0]]
            bboxes_lbs = p_dict[1][provided_ptypes[0] + '_labels']

            for box in bboxes:
                if not any(box[0, i] == box[0, i+3] for i in range(3)):
                    warnings.warn('nnInteractive natively supports 2D bounding boxes, received a 3D bounding box in the request!')
            #Now we convert the bounding boxes to the expected format, which is to have the bboxes represented by a half-open interval. As opposed to the 
            #closed interval representation used in the API. The upper bound is the open end.
            temp_bboxes = torch.cat(bboxes, dim=0)
            temp_bboxes[:, 3:] += 1 #Converting to half-open interval representation by adding 1 to the upper bounds.
            #Clamping the bounding boxes to be within the image dimensions.
            temp_bboxes[:, :3] = torch.max(torch.zeros(temp_bboxes.shape[0], 3), temp_bboxes[:, :3])
            temp_bboxes[:, 3:] = torch.min((torch.tensor(self.session.original_image_shape[1:]) - 1).unsqueeze(0).repeat(temp_bboxes.shape[0], 1), temp_bboxes[:, 3:])
            #Clamping above from the index! not the shape itself. 

            #Now we will vectorise the process of converting to the expected format.
            #Expected structure of the bboxes for nninteractive input is a list: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            converted_bboxes = torch.stack(
                [temp_bboxes[:,::3], temp_bboxes[:,1::3], temp_bboxes[:,2::3]], dim=1
                ).tolist()  
            #This is very convoluted looking... I know. But it works. The input was structured as [num_bboxes, 6] where the 6 represents:
            # [x_min, y_min, z_min, x_max, y_max, z_max]. We want to convert this to the expected format of nnInteractive. We first stack to reshape
            # the 6 values into 3 pairs. We then convert to list, this creates a nested list of bboxes. 
            
            #Each bbox will have structure [[x_min, x_max], [y_min, y_max], [z_min, z_max]] now, as expected. 

            #Lets handle the bbox labels, first we will make the common sense restriction that each class can only have 1 bounding box per callback. 
            # Not even the most restrictive case (i.e., only one class can have an interaction). 
            
            bin_counts = torch.bincount(torch.stack(bboxes_lbs).flatten())
            if any([count > 1 for count in bin_counts]):
                raise Exception('Each class can only have one bounding box prompt at a given time! Cannot proceed with interactive inference!')
            if bin_counts.shape[0] > 2:
                raise Exception('More than two class labels were provided bounding box prompts OR bbox label was outside of the [0,1] range! Cannot proceed with interactive inference!') 
            
            #First we will look at the background class, then the foreground class, because we want the center of the last interaction to be the foreground class.
            bg_code = self.configs_labels_dict['background']
            
            if bg_code != 0:
                raise Exception('Script written assuming background is assigned class 0! Cannot proceed with inference!')
            if bg_code in bboxes_lbs:
                bg_idx = (torch.cat(bboxes_lbs) == bg_code).nonzero(as_tuple=True)
                if len(bg_idx[0]) > 1:
                    raise Exception('Each class can only have one bounding box prompt at a given time! Cannot proceed with interactive inference!')
                bg_idx = bg_idx[0].item()
                self.session.add_bbox_interaction(
                    converted_bboxes[bg_idx],
                    include_interaction=False,
                    run_prediction=False
                )
            fg_code = self.configs_labels_dict[[k for k in self.configs_labels_dict.keys() if k != 'background'][0]]
            if fg_code in bboxes_lbs:
                fg_idx = (torch.cat(bboxes_lbs) == fg_code).nonzero(as_tuple=True)
                if len(fg_idx[0]) > 1:
                    raise Exception('Each class can only have one bounding box prompt at a given time! Cannot proceed with interactive inference!')
                fg_idx = fg_idx[0].item()
                self.session.add_bbox_interaction(
                    converted_bboxes[fg_idx],
                    include_interaction=True,
                    run_prediction=False
                )
    
        elif provided_ptypes[0].title() == "Lasso":
            raise NotImplementedError('Conversion from api-structure to array form not yet implemented')
        else:
            raise Exception('No other prompting subtypes are supported in nnInteractive.')



    def binary_predict(self):
        '''
        bbox: list[dict] | None,
        clicks: list[dict] | None,
        clicks_order: list[list[str]] | None,
        prev_pred: np.ndarray | None,
        '''
        # now run inference on the last interaction center
        self.session.new_interaction_centers = [self.session.new_interaction_centers[-1]]
        self.session.new_interaction_zoom_out_factors = [self.session.new_interaction_zoom_out_factors[-1]]
        self.session._predict()
        pred = self.session.target_buffer.unsqueeze(0) #Adding back the batch dimension..., we don't assume a one-hot format.
        # del session #We don't delete the session here because we want to keep the application online..
        empty_cache(torch.device('cuda', 0))
        probs_tensor = torch.zeros([2] + list(self.session.target_buffer.shape), dtype=torch.float32) #This is a dummy..they don't give us this. Also its probably going to be deprecated soon, but has not been yet. So just put a dummy.

        return pred, probs_tensor


    def binary_subject_prep(self, request:dict):
        self.dataset_info = request['dataset_info']
        if len(self.dataset_info['task_channels']) != 1:
            raise Exception('The inference app only supports single channel images for segmentation.')
        
        if request['infer_mode'] == 'IS_interactive_edit':
            is_state = request['i_state']
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')
            init = False 

        elif request['infer_mode'] == 'IS_interactive_init':
            is_state = request['i_state']
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')

            init = True
            self.configs_labels_dict = request['config_labels_dict']
            self.load_new_image(request['image']['metatensor'])
            self.session.reset_interactions()
            # self.prev_pred = None  We don't need this. The buffer is already reset.
            empty_cache(torch.device('cuda', 0))
            
        elif request['infer_mode'] == 'IS_autoseg':            
            raise Exception('True Autoseg is too OOD for this algorithm (i.e. not the S.A.T but truly without any prompts)')

        affine = request['image']['meta_dict']['affine']

        return init, affine, is_state 

    def load_new_image(self, image: torch.Tensor):
        self.session.set_image(image.numpy().astype(np.float32))
        target_buffer = torch.zeros(image.shape[1:], dtype=torch.uint8, device='cpu')
        self.session.set_target_buffer(target_buffer)
    
    def __call__(self, request:dict):

        if len(request['config_labels_dict']) == 2:
            class_type = 'binary'
        elif len(request['config_labels_dict']) > 2:
            class_type = 'multi'
            raise NotImplementedError('See the SegFM implementation for integrating multi-class segmentation interpretation.')
        else:
            raise Exception('Should not have received less than two class labels at minimum')
        
        #We create a duplicate so we can transform the data from metatensor format to the torch tensor format compatible with the inference script.
        modif_request = copy.deepcopy(request) 

        app = self.infer_apps[modif_request['infer_mode']][f'{class_type}_predict']

        #Setting the configs label dictionary for this inference request.
        self.configs_labels_dict = modif_request['config_labels_dict']


        pred, probs_tensor, affine = app(request=modif_request)

        pred = pred.to(device='cpu')
        probs_tensor = probs_tensor.to(device='cpu')
        # affine = affine.to(device='cpu')
        torch.cuda.empty_cache()

        assert probs_tensor.shape[1:] == request['image']['metatensor'].shape[1:]
        assert pred.shape[1:] == request['image']['metatensor'].shape[1:] 
        assert torch.all(affine == request['image']['meta_dict']['affine'])
        assert isinstance(probs_tensor, torch.Tensor) 
        assert isinstance(pred, torch.Tensor)
        assert isinstance(affine, torch.Tensor)

        output = {
            'probs':{
                'metatensor':probs_tensor,
                'meta_dict':{'affine': affine}
            },
            'pred':{
                'metatensor':pred,
                'meta_dict':{'affine': affine}
            },
        }
        #Functionally probably wont do anything but putting it here as a placebo. Won't make a diff because there are references
        #to these variables throughout.
        del pred 
        del probs_tensor
        del affine
        del modif_request
        # torch.cuda.empty_cache() 
        empty_cache(torch.device('cuda', 0))

        return output





#NOTE: The following is the original inference function for multi-class segmentation (well, in their words object/instance....).  

# for oid in range(1, num_objects + 1):
#             # place previous segmentation
#             if prev_pred is not None:
#                 session.add_initial_seg_interaction((prev_pred == oid).astype(np.uint8), run_prediction=False)
#             else:
#                 session.reset_interactions()
#             if bbox is not None:
#                 bbox_here = bbox[oid - 1]
#                 bbox_here = [
#                     [bbox_here['z_min'], bbox_here['z_max'] + 1],
#                     [bbox_here['z_mid_y_min'], bbox_here['z_mid_y_max'] + 1],
#                     [bbox_here['z_mid_x_min'], bbox_here['z_mid_x_max'] + 1]
#                     ]
#                 session.add_bbox_interaction(bbox_here, include_interaction=True, run_prediction=False)
#             if clicks is not None:
#                 clicks_here = clicks[oid - 1]
#                 clicks_order_here = clicks_order[oid - 1]
#                 fg_ptr = bg_ptr = 0
#                 for kind in clicks_order_here:
#                     if kind == 'fg':
#                         click = clicks_here['fg'][fg_ptr]
#                         fg_ptr += 1
#                     else:
#                         click = clicks_here['bg'][bg_ptr]
#                         bg_ptr += 1

#                     print(f"Class {oid}: {kind} click at {click}")
#                     session.add_point_interaction(click, include_interaction=kind == 'fg', run_prediction=False)
#             # now run inference on the last interaction center
#             session.new_interaction_centers = [session.new_interaction_centers[-1]]
#             session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
#             session._predict()
#             result[session.target_buffer > 0] = oid

#         # del session #We don't delete the session here because we want to keep the application online..
#         empty_cache(torch.device('cuda', 0))
#         return result.cpu().numpy()

if __name__ == '__main__':
   
    infer_app = InferApp(
        infer_device=torch.device('cuda', index=0)
        )

    infer_app.app_configs()

    from monai.transforms import LoadImaged, Orientationd, EnsureChannelFirstd, Compose 
    import nibabel as nib 

    input_dict = {
        'image' :os.path.join(app_local_path, 'debug_image/BraTS2021_00266.nii.gz')
        }    
    load_and_transf = Compose([LoadImaged(keys=['image'], image_only=True), EnsureChannelFirstd(keys=['image']), Orientationd(keys=['image'], axcodes='RAS')])

    loaded_im = load_and_transf(input_dict)
    input_metatensor = torch.from_numpy(loaded_im['image'].array)
    meta = {
        'original_affine': copy.deepcopy(torch.from_numpy(loaded_im['image'].meta['original_affine']).to(dtype=torch.float64)), 
        'affine': copy.deepcopy(loaded_im['image'].meta['affine']).to(dtype=torch.float64)}
    
    request = {
        'image':{
            'metatensor': input_metatensor,
            'meta_dict':meta
        },
        # 'infer_mode':'IS_interactive_edit',
        'infer_mode': 'IS_interactive_init',
        'config_labels_dict':{'background':0, 'tumor':1},
        'dataset_info':{
            'dataset_name':'BraTS2021_t2',
            'dataset_image_channels': {            
                "T2w": "0"
            },
            'task_channels': ["T2w"]
        },
        'i_state':
            {
            'interaction_torch_format': {
                'interactions': {
                    'points': None, #[torch.tensor([[40, 103, 43]]), torch.tensor([[61, 62, 39]])], #None 
                    'scribbles': None, 
                    'bboxes': [torch.Tensor([[56,30,17, 92, 76, 51]]).to(dtype=torch.int64)] #None 
                    },
                'interactions_labels': {
                    'points_labels': None, # [torch.tensor([0]), torch.tensor([1])], #None,
                    'scribbles_labels': None, 
                    'bboxes_labels': [torch.Tensor([1]).to(dtype=torch.int64)] #None
                    }
                },
            'interaction_dict_format': {
                'points': None, 
                # {
                    # 'background': [[40, 103, 43]],
                    # 'tumor': [[61,62,39]]
                    #},  
                'scribbles': None,
                'bboxes': {'background': [], 'tumor': [[56,30,17, 92, 76, 51]]} #None
                },    
        },
    }
    output = infer_app(request)
    print('halt')
