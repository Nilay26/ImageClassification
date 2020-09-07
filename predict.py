exec(compile(open('importLibraries.py', "rb").read(), 'importLibraries.py', 'exec'))

arg_parser = argparse.ArgumentParser()
default_testimage = './flowers/test/89/image_00699.jpg'
default_checkpoint = './checkpoint.pth'

def flowernames_from_json(jsonfile):
    with open(jsonfile, 'r') as file:
        map_flowername = json.load(file)
        #print(map_flowername)
    return map_flowername

def get_flowername(val): 
    for key, value in flowernames.items(): 
         if val == key: 
             return value
        
arg_parser.add_argument('--test_path', dest="test_path", action='store', type=str, help= 'Path of the image to be tested', default=default_testimage)
arg_parser.add_argument('--checkpoint', dest="checkpoint", action='store', type=str, help= 'Path of the saved model', default=default_checkpoint)
arg_parser.add_argument('--top_k', dest="top_k", action='store', type=int, help= 'Number of top predicted classes of flowers', default=5)
arg_parser.add_argument('--jsonfile', dest="jsonfile", action='store', type=str, help= 'Path of the json file', default='cat_to_name.json')
arg_parser.add_argument('--device', dest="device", action="store", default = 'cuda' , type = str, help="Select GPU name or CPU to choose the device. By default, cuda")
model = load_model.load_checkpoint(arg_parser.parse_args().checkpoint, arg_parser.parse_args().device)
flowernames = flowernames_from_json(arg_parser.parse_args().jsonfile)

top_k_probabilities, top_k_classes = model_prediction.testdata_prediction(arg_parser.parse_args().test_path, model, arg_parser.parse_args().top_k, arg_parser.parse_args().device)
#print(top_k_probabilities)
#print(top_k_classes)
#print(flowernames)
#torch.stack(top_k_probabilities,top_k_classes, dim=0, out=None)

topk_list =  np.arange(0, arg_parser.parse_args().top_k, 1).tolist()
for i in topk_list:
    print("Flower Name: {} <<<-->>> Probability:{}".format(flowernames[top_k_classes[i]],top_k_probabilities[0][i]))
