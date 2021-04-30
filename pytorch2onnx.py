
import torch

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

#pth转onnx
def model_convert():

    model=torch.load('./model/model_epoch_100.pth')["model"] 
    model.to(device)
    model.eval()

    dummy_input=torch.randn(1,1,240,240,device=device)  
    input_names=['input']  
    output_names=['output_x2','output_x4']  
    torch.onnx.export(model,dummy_input,'./results/lapsrn.onnx',export_params=True,
                        verbose=True,
                        input_names=input_names,
                        output_names=output_names)

if __name__=="__main__":
    model_convert()
