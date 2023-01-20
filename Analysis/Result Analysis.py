import os
import pandas as pd

current = os.getcwd()
lst = [ele for ele in os.listdir(current)]
for topic in lst[:6]:
    sublst = [ele for ele in os.listdir(os.path.join(current, topic))]
    for model in sublst:
        files = [ele for ele in os.listdir(os.path.join(current, topic, model)) if 'report' in ele]
        score_lst = []
        for file in files:
            result = pd.read_csv(os.path.join(current, topic, model,file))
            score_lst.append(result.loc[2]['weighted avg'])
        print(score_lst)
        try:
            avg = sum(score_lst)/len(score_lst)
            score_lst.insert(0, avg)
            final_lst = [str(ele)[:8] for ele in score_lst]
            with open(os.path.join(current, topic, model, 'result.txt'), 'w') as f:
                for line in final_lst:
                    f.write(f"{line}\n")
        except ZeroDivisionError:
            pass



