exec(compile(open('importLibraries.py', "rb").read(), 'importLibraries.py', 'exec'))

#Build out the neural layers

def build_layer(architecture, hiddenlayer_units, learning_rate, device):
           
    if architecture == 'vgg16':
       model = models.vgg16(pretrained=True)
       # Freeze parameters so we don't backprop through them
       for param in model.parameters():
           param.requires_grad = False
       classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088,hiddenlayer_units)),
                                                ('relu1', nn.ReLU()),
                                                ('dropout1', nn.Dropout(p=0.5)),                                                                                       ('fc2', nn.Linear(hiddenlayer_units,1024)),
                                                ('relu2', nn.ReLU()),
                                                ('dropout2', nn.Dropout(p=0.5)),  
                                                ('fc3', nn.Linear(1024,102)),
                                                ('output', nn.LogSoftmax(dim=1))]))

       model.classifier = classifier
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024,hiddenlayer_units)),
                                                ('relu1', nn.ReLU()),
                                                ('dropout', nn.Dropout(p=0.5)),
                                                ('fc2', nn.Linear(hiddenlayer_units,512)),
                                                ('relu2', nn.ReLU()),
                                                ('dropout', nn.Dropout(p=0.5)),
                                                ('fc3', nn.Linear(256,102)),
                                                ('output', nn.LogSoftmax(dim=1))]))

        model.classifier = classifier
    else:
        print("The model you requested for training is not currently supported. Please enter either vgg16 or                   densenet121.")
    
      
    
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    model.to(device);
    
    return model, criterion, optimizer

    