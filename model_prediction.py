exec(compile(open('importLibraries.py', "rb").read(), 'importLibraries.py', 'exec'))

def testdata_prediction(image_path, model, topk, device): 

    image = Image.open(image_path)
    image_augmentation = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_augmented = image_augmentation(image)
    image_augmented = image_augmented.unsqueeze_(0)
    image_augmented = image_augmented.float()
    
    
    gpu_flag = torch.cuda.is_available()
    if gpu_flag:
        model.to(device)
        with torch.no_grad():
            model.eval()
            test_result = model.forward(image_augmented.to(device))
            probability = torch.exp(test_result)
    else:
        with torch.no_grad():
            test_result = model.forward(image_augmented)
            probability = torch.exp(test_result)
    
    top_k_probabilities, top_k_index = torch.topk(probability, topk)
    
    index_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_k_classes = list(map(lambda index: index_to_class[index], np.array(top_k_index.cpu())[0]))
    #top_k_classes = np.array(top_k_classes, dtype=np.int)
    #torch.FloatTensor(top_k_classes)
    return top_k_probabilities, top_k_classes
