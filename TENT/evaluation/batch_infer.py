import torch
def get_result(list):
    flattened_list = [item for sublist in list for item in sublist]
    data_count=len(flattened_list)
    correct_result=0
    for item in flattened_list:
        if(item['predicted']==item['real_score']):
            correct_result+=1
    return {
         "result_list":flattened_list,
         "correct_result":correct_result,
         "accuracy": correct_result/data_count
    }

def predict_image( model,x,y):
    with torch.no_grad():
        x=x.to(torch.device("cuda"))
        pred = model(x)  
        _, preds = torch.max(pred, dim=1)  

        result = [] 

        for i in range(len(pred)):
            result.append({
            "predicted": preds[i].item(),
            "predicted_list": pred[i].tolist(),
            "real_score": y[i].item()
        })
        return result
    

def infer_by_batch(model, dl):
    
    results=[]
    model.eval()
    for x, y,_ in dl:
            result= predict_image(model,x,y)
            results.append(result)
    return results

def infer_by_batch_split(model, dl_train,dl_test):
    
    results_train=[]
    results_test=[]
    model.eval()
    for x, y,_ in dl_train:
            result= predict_image(model,x,y)
            results_train.append(result)
    for x, y,_ in dl_test:
            result= predict_image(model,x,y)
            results_test.append(result)
    return results_train,results_test