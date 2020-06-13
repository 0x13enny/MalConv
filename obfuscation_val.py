

def test_model(weight_dir):
    first_n_byte = 2000000
    num_workers = 2
    # batch_size = 32
    batch_size = 1
    model = model_MalConv.MalConv()
    rank = 0
    world_size = 2
    device = utils.model_to_cuda(model)

    # with open('labels/test_path.csv', newline='') as f:
    with open('labels/malware.csv', newline='') as f:
       reader = csv.reader(f)
       test_set = list(reader)
    test_loader = DataLoader(PE_Dataset(test_set, first_n_byte),
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)

    with open('labels/o_malware.csv', newline='') as f:
       reader = csv.reader(f)
       o_test_set = list(reader)
    o_test_loader = DataLoader(PE_Dataset(o_test_set, first_n_byte),
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)
    try:

        assert os.path.exists(weight_dir)
        model_dir = weight_dir
        state = torch.load(model_dir,map_location=device)
        model.load_state_dict(state)

        model.eval()
        TPR = []
        FPR = []
        THRESHOLD = []
        scores = []
        o_scores = []
        with torch.no_grad():
            count = 0
            for batch_data, label in o_test_loader:
                if device is not None:
                    batch_data, label = batch_data.to(device), label.to(device)
                output = model(batch_data)
                # print(label)
                label = label.cpu().numpy()[:,1]
                score = output.cpu().numpy()[:,1]

                o_scores.append(score)
                # y_pred = [1 if y>0.5 else 0 for y in scores]
            print("--------------------------------")

            count = 0
            for batch_data, label in test_loader:
                if device is not None:
                    batch_data, label = batch_data.to(device), label.to(device)
                output = model(batch_data)
                # print(label)
                label = label.cpu().numpy()[:,1]
                score = output.cpu().numpy()[:,1]

                scores.append(score)
                y_pred = [1 if y>0.5 else 0 for y in scores]
                # fpr, tpr, threshold = roc_curve(label, y_pred)
                # TPR += tpr
                # FPR += fpr
                # THRESHOLD += threshold
                
                # loss = F.cross_entropy(output, label)
                # print(output)
                # sys.exit(1)
            result = zip(scores, o_scores)
            with open("result.csv",'w') as f:
                count = 0
                writer = csv.writer(f)
                writer.writerow(['sha256', 'score of original malware', 'score after obfuscation'])
                for s, o_s in result:
                    writer.writerow([test_set[count][0].split('/')[-1], s[0], o_s[0]])
                    count +=1
        # roc_auc = auc(FPR, TPR)
        # print(roc_auc)

    except AssertionError:
        print("No model")
        
    # with open('labels/mal_test_with_score.csv',"w") as f:
    #     for i in range(len(test_set)):
    #         writer = csv.writer(f)
    #         writer.writerow([test_set[i][0], scores[0][i]])
