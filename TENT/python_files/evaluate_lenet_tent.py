import tent 
import torch 
from torch.utils.data import DataLoader
import importlib
# import evaluate_lenet_tent
# import batch_data_loader
import batch_infer
importlib.reload(batch_infer)  
from batch_infer import infer_by_batch,get_result, infer_by_batch_split



def evaluate_models_base_only(models, dataloader):
    results = {}
    for name, model in models.items():
        # Ensure model is in evaluation mode
        model.eval()
        
        # Perform batch inference without TENT
        result = infer_by_batch(model, dataloader)
        accuracy_no_tent=get_result(result)['accuracy']
        results[name] = {
            "accuracy_no_tent": accuracy_no_tent
            }
    
    return results




def evaluate_models(models, dataloader, steps=1,isTentedModelsOnly=False):


    results = {}
    for name, model in models.items():
 
        print(f"Currently on model name {name}")
        model.eval()
        if isTentedModelsOnly==False:
        
            result = infer_by_batch(model, dataloader)
            accuracy_no_tent=None
            accuracy_no_tent = get_result(result)['accuracy'] 
        

        tented_model = tent.configure_model(model)
        params, _ = tent.collect_params(tented_model)
        optimizer = torch.optim.Adam(params, lr=1e-3)
        tented_model = tent.Tent(tented_model, optimizer, steps=steps)
        result_tented = infer_by_batch(tented_model, dataloader)
        accuracy_tented = get_result(result_tented)['accuracy']
    
        if isTentedModelsOnly==False:
            results[name] = {
            "accuracy_no_tent": accuracy_no_tent,
            "accuracy_tented": accuracy_tented
            }
        else:
             results[name] = {
            "accuracy_tented": accuracy_tented
            }
    
    return results



def evaluate_models_test_traindl(models,train_dl,test_dl, steps=1,isTentedModelsOnly=False):
    results = {}
    for name, model in models.items():
        print(f"Currently on model name {name}")
        model.eval()
        if isTentedModelsOnly==False:

            result= infer_by_batch_split(model, train_dl,test_dl)
            accuracy_no_tent=None
            accuracy_no_tent = get_result(result)['accuracy']
        
        
        tented_model = tent.configure_model(model)
        params, _ = tent.collect_params(tented_model)
        optimizer = torch.optim.Adam(params, lr=1e-3)
        tented_model = tent.Tent(tented_model, optimizer, steps=steps)
    
        results_train,results_test= infer_by_batch_split(tented_model, train_dl,test_dl)
        accuracy_train = get_result(results_train)['accuracy']
        accuracy_test = get_result(results_test)['accuracy']
        if isTentedModelsOnly==False:
            results[name] = {
            "accuracy_no_tent": accuracy_no_tent,
            "accuracy_tented_train": accuracy_train,
            "accuracy_tented_test": accuracy_test

            }
        else:
             results[name] = {
            "accuracy_tented_train": accuracy_train,
            "accuracy_tented_test": accuracy_test
            }
    
    return results