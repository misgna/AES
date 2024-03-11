import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score
from utilities import min_max_normalization, scaler, essay_dataloader
from mlp import MLP

def training(model, optimizer, data, criterion, prompt):
    model.train()
    loss = 0.0
    for (essay, scores) in data:
        optimizer.zero_grad()
        
        output = model(essay)
        scores = torch.tensor([min_max_normalization(score.item(), prompt) for score in scores], dtype=torch.float32)
        scores = scores.reshape(-1, 1)
        loss = criterion(output, scores)
        loss.backward()
        optimizer.step()

def testing(model, data, criterion, prompt):
    model.eval()
    total_loss = 0.0
    scores_4_qwk, output_4_qwk = [], []
    with torch.no_grad():
        for (essay, scores) in data:
            scores = scores.reshape(-1, 1)
            output = model(essay)
            output = torch.tensor([scaler(out.item(), prompt) for out in output ], dtype=torch.float32)
            output = output.reshape(-1, 1)
            loss = criterion(output, scores)
            total_loss += loss
            
            scores_4_qwk.append(scores)
            output_4_qwk.append(output)
            
    score_list = [score.item() for tensor_score in scores_4_qwk for score in tensor_score]
    output_list = [int(output.item()) for tensor_output in output_4_qwk for output in tensor_output]
    
    qwk = cohen_kappa_score(score_list, output_list, weights='quadratic')
    return qwk, total_loss / len(data)

#prompt = 1
embedding_dim = 50
criterion = nn.MSELoss()
hidden_dim = 100
EPOCHS = 50
batch_size = 32
num_classes = 1

qwk_prompts_val, mse_prompts_val, qwk_prompts_test, mse_prompts_test  = [], [], [], []

for prompt in range(1, 9):
    train_dl, val_dl, test_dl, vocab_essay = essay_dataloader(prompt, batch_size)
    vocab_size = len(vocab_essay)
    
    model = MLP(vocab_size, embedding_dim, hidden_dim, num_classes)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
    
    qwk_epoch, mse_epoch = [], []
    for epoch in range(0, EPOCHS):
        training(model, optimizer, train_dl, criterion, prompt) # Training
        qwk, mse = testing(model, val_dl, criterion, prompt) # Validation
        
        qwk_epoch.append(qwk)
        mse_epoch.append(mse.item())
        
    qwk_prompts_val.append(qwk_epoch)
    mse_prompts_val.append(mse_epoch)

    #Testing
    qwk_test, mse_test = testing(model, test_dl, criterion, prompt)
    qwk_prompts_test.append(qwk_test)
    mse_prompts_test.append(mse_test) 

for idx, qwk in enumerate(qwk_prompts_test):
    print(f"Prompt {idx + 1}, QWK: {qwk}")

print(f"Avg. QWK: {sum(qwk_prompts_test)/len(qwk_prompts_test)}")