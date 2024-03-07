import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os


def train_val(loader_train, loader_val, model, criterion, optimizer, epochs, print_every, device,
              scheduler,
              output_dir=None):
    best_val = 0
    train_loss = 0
    val_loss = 0
    train_correct = 0
    train_total = 0
    val_correct = 0
    val_total = 0
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(loader_train):
            model.train()
            inputs, targets = inputs.to(device), targets.to(device)
            #######################################################################
            # TODO: Implement training of LeNet.                                  #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            if (batch_idx + 1) % print_every == 0:
                print('Epoch: %d | Batch: %d/%d | Loss: %.3f | Acc: %.3f%%'
                      % (epoch + 1, batch_idx + 1, len(loader_train), train_loss / (batch_idx + 1), 100. * train_correct / train_total))
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #######################################################################
            #                           END OF YOUR CODE                          #
            #######################################################################
        scheduler.step()
        with torch.no_grad():
            for val_idx, (inputs, targets) in enumerate(loader_val):
                model.eval()
                inputs, targets = inputs.to(device), targets.to(device)
                #######################################################################
                # TODO: Implement testing of LeNet.                                   #
                #######################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                #######################################################################
                #                           END OF YOUR CODE                          #
                #######################################################################
            print('val Loss: %.3f | Acc: %.3f%%' % (val_loss / (val_idx + 1), 100. * val_correct / val_total))
        # Adjust learning rate
        if scheduler is not None:
            scheduler.step()
        # Save checkpoint.
        acc = 100. * val_correct / val_total
        if acc > best_val and output_dir is not None:
            print('Saving model...')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            # Delete the previous best model file if it exists
            previous_best_model_path = os.path.join(output_dir, 'best_model.pth')
            if os.path.exists(previous_best_model_path):
                os.remove(previous_best_model_path)
            torch.save(state, previous_best_model_path)
            best_val = acc
        train_loss = 0
        val_loss = 0
    return


def predict(test_loader, model, criterion, device):
    r"""
    You can save the model file in output_ Under dir
    """
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()
            inputs, targets = inputs.to(device), targets.to(device)
            #######################################################################
            # TODO: Implement testing of LeNet.                                   #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #######################################################################
            #                           END OF YOUR CODE                          #
            #######################################################################
        print('testing Loss: %.3f | Acc: %.3f%%' % (test_loss / (batch_idx + 1), 100. * correct / total))

# model	layer	active function 	parameters	     output size
# model1	conv1	     tanh	        c=6,k=5,s=1,p=0	    6*28*28
# 	    maxpool1		            k=2,s=2	            6*14*14
# 	    conv2		                c=16,k=5,s=1,p=0	16*10*10
# 	    maxpool2	    	        k=2,s=2	            16*5*5
# 	    fc1	    	                in=-1,out=120	
# 	    fc2		                    in=120,out=84	
# 	    fc3		                    in=84,out=10	

# model	layer	active function 	parameters	     output size     
# model2	conv1	    relu	        c=96,k=7,s=1,p=3	96*32*32
# 	    maxpol1		                k=3,s=2	            96*15*15
# 	    lrn1		                f=96	
# 	    conv2		                c=128,k=5,s=1,p=2	128*15*15
# 	    maxpol2		                k=3,s=2	            128*7*7
# 	    lrn2		                f=128	
# 	    conv3		                c=256,k=3,s=1,p=1	256*7*7
# 	    maxpool3		            k=3,s=2	            256*3*3
# 	    lrn3		                f=256	
# 	    fn1		                    in=-1,out=4096	
# 	    dropout		                p=0.5	
# 	    fn2		                    in=4096,out=4096	
# 	    dropout		                p=0.5	
# 	    fn3		                    in=4096,out=10	

# model	layer	active function 	parameters	     output size    
# model3	conv11	    relu	        c=64,k=3,s=1,p=1	64*32*32
# 	    bn1		                    f=64	
# 	    conv12		                c=64.k=3,s=1,p=1	64*32*32
# 	    bn2		                    f=64	
# 	    maxpool1		            k=2,s=2	            64*16*16
# 	    conv21		                c=128,k=3,s=1,p=1	128*16*16
# 	    bn1		                    f=128	
# 	    conv22		                c=128,k=3,s=1,p=1	128*16*16
# 	    bn2		                    f=128	
# 	    conv23		                c=128,k=3,s=1,p=1	128*16*16
# 	    bn3		                    f=128	
# 	    maxpool2		            k=2,s=2	            128*8*8
# 	    conv31		                c=256,k=3,s=1,p=1	256*8*8
# 	    bn1		                    f=256	
# 	    conv32		                c=256,k=3,s=1,p=1	256*8*8
# 	    bn2		                    f=256	
# 	    conv33		                c=256,k=3,s=1,p=1	256*8*8
# 	    bn3		                    f=256	
# 	    maxpool3		            k=2,s=2	            256*4*4
# 	    conv41		                c=512,k=3,s=1,p=1	
# 	    bn1		                    f=512	
#     	conv42		                c=512,k=3,s=1,p=1	
#         bn2		                    f=512	
# 	    conv43		                c=512,k=3,s=1,p=1	
# 	    bn3		                    f=512	
#     	maxpool		                k=2,s=2	
#     	fn1		                    in=-1,out=4090	
# 	    dropout		                p=0.5	
# 	    fn2		                    in=4090,out=1000	
# 	    dropout		                p=0.5	
# 	    fn3		                    in=1000,out=10	