exec(compile(open('importLibraries.py', "rb").read(), 'importLibraries.py', 'exec'))
from workspace_utils import active_session

with active_session():
    def train_data_nn(device, trainloader, epochs, model, optimizer, criterion, validloader):
        criterion = criterion
        validloader = validloader
        model = model
        gpu_flag = torch.cuda.is_available()
        print(gpu_flag)
        if gpu_flag:
            model.to(device)
        else:
            print("Device not available")

        print_step = 20
        steps = 0
        print("Training Started")
        for e in range(epochs):
            run_loss = 0
            for images, labels in iter(trainloader):
                if gpu_flag:
                    images = images.to(device)
                    labels = labels.to(device)
                else:
                    print("Device not available")
                    break;
                steps += 1
                optimizer.zero_grad()
                #print("Forward and Backward passes")
                #Forward and Backward passes
                #print(images)
                #print(labels)
                output = model.forward(images)
                loss = criterion(output, labels)
                #print(output)
                #print(loss)
                loss.backward()
                optimizer.step()

                run_loss += loss.item()

                

                if steps % print_step == 0:
                    model.eval()
                    valid_loss, accuracy = validate_prediction(device, model, validloader, criterion, gpu_flag, optimizer)
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Training Loss: {:.3f}".format(run_loss/print_step),
                          "Validation_Loss: {:.3f}".format(valid_loss),
                          "Accuracy: {:.3f}".format(accuracy))    


                    run_loss = 0              

    # Implement a function for the validation pass
    def validate_prediction(device, model, loader, criterion, gpu_flag, optimizer):
        for images, labels in iter(loader):

            if gpu_flag:
                images,labels= images.to(device) , labels.to(device)
                model.to(device)           
            optimizer.zero_grad()
            #print(gpu_flag)
            loss = 0
            accuracy = 0


            output = model.forward(images)
            loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()    

        return loss, accuracy    


    # Save checkpoint

    def save_model(model, architecture, hiddenlayer_units, learning_rate, model_path, training_data):

        model.to('cpu')
        
        model.class_to_idx = training_data.class_to_idx
        checkpoint = {'architecture': architecture,
                  'hiddenlayer_units': hiddenlayer_units,
                  'learning_rate': learning_rate,
                  'class_to_idx':model.class_to_idx,
                  'state_dict': model.state_dict()}

        #torch.save(checkpoint, model_path)
        print("Model Saved")
    
    