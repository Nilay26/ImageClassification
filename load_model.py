exec(compile(open('importLibraries.py', "rb").read(), 'importLibraries.py', 'exec'))
# Loading saved model

def load_checkpoint(filepath, device):
    if device == 'cuda' and torch.cuda.is_available():
        print("Prediction running on GPU")
    elif device == 'cuda' and torch.cuda.is_available() == False:
        device = 'cpu'
        print("GPU unavailable. Enable GPU. Prediction running on CPU")
    else:
        print("Only GPU available is CUDA. Kindly use cuda to proceed")
        sys.exit(0)
        
    checkpoint = torch.load(filepath)
    model, criterion, optimizer = build_Classifier.build_layer(checkpoint['architecture'],                                 checkpoint['hiddenlayer_units'], checkpoint['learning_rate'], device)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    #print(model)
    return model