```python
# before training(
misses = []
classifier_layers = [l1, l2, l3, l_op]
datasets = []
#)

for experience in train_stream:
    datasets.append((experience.dataset))
    
    for epoch in range(experience.epochs):
        for batch in experience.loader:
            x, y = batch
            y_preds = Model(x)
            
            #optimizer steps
            
    # after training experience(
    super_dataset = make_dataset(make_dataset(datasets))
    x, y, y_preds, act_ls = do_inference(super_dataset)
    misses, hits = seperate_misses(x, y, y_preds, act_ls)
    
    adjustable_layer_to_point_map = calculate_fixes(misses)
    
    loss = fix_loss(misses) + c * preservation_regularization(hits)
    
    # optimization step
    
    
    
    
    
    
    
```