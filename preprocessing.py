from preprocess.preprocessing_h2o import (
    preprocessing_object as pp_obj_h2o, 
    preprocessing_data as pp_data_h2o, 
    preprocessing_text as pp_text_h2o, 
    preprocessing_balance_weights as pp_bw_h2o, 
    preprocessing_text2length as pp_t2l_h2o, 
    print_text_data_num as print_h2o, 
)
from preprocess.preprocessing_grab import (
    preprocessing_object as pp_obj_grab, 
    preprocessing_data as pp_data_grab, 
    preprocessing_text as pp_text_grab, 
    preprocessing_balance_weights as pp_bw_grab, 
    preprocessing_text2length as pp_t2l_grab, 
    print_text_data_num as print_grab, 
)
from preprocess.preprocessing_arctic import (
    preprocessing_object as pp_obj_arctic, 
    preprocessing_data as pp_data_arctic, 
    preprocessing_text as pp_text_arctic, 
    preprocessing_balance_weights as pp_bw_arctic, 
    preprocessing_text2length as pp_t2l_arctic, 
    print_text_data_num as print_arctic, 
)

if __name__ == '__main__':
    pp_obj_h2o()
    pp_data_h2o()
    pp_text_h2o()
    pp_bw_h2o()
    pp_t2l_h2o()
    print_h2o()
    
    pp_obj_grab()
    pp_data_grab()
    pp_text_grab()
    pp_bw_grab()
    pp_t2l_grab()
    print_grab()
    
    pp_obj_arctic()
    pp_data_arctic()
    pp_text_arctic()
    pp_bw_arctic()
    pp_t2l_arctic()
    print_arctic()