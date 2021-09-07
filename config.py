
def get_config():
    sweep_config = {
        'method': 'random'
        }

    metric = {
        'name': 'best_val_loss',
        'goal': 'minimize'   
        }

    sweep_config['metric'] = metric
    parameters_dict = {
        'epochs': {
            'values': [5, 10]
            },
        'lr': {
            'values': [0.0007, 0.001]
            },
        'batch_size': {
            'values': [32, 64, 128]
            },
        'wd': {
            'values': [ 0, 0.000001, 0.00001, 0.0001, 0.001]
            },
        'model_arch': {
            'values': [ 'rexnet1_0x', 'rexnet1_3x']
            },
        'image_size': {
            'value': 224
            }

            
        }

    sweep_config['parameters'] = parameters_dict
    parameters_dict.update({
        'classes': {
            'value': 1}
        ,
        'freeze': {
            'value': None}
        ,
        'train_folder': {
            'value': 'DS_iqa/train'}
        ,
        'test_folder': {
            'value': 'DS_iqa/test'}
        ,
        
        'checkpoint': {
            'value': 'checkpoint.pth'}
    }) 

    return sweep_config
