import json

'''
Merge the three JSON files into a single one, removing the passage entry.
Some questions are specific to some part of the story, and thus must be adjusted afterwards, since the passage is discarded.
'''
dataset = []
for file in [(0, 'yes'), (1, 'irrelevant'), (2, 'no')]:
    with open(f'dataset/question/{file[1]}.json', "r") as f:
        data = list(map(lambda e: {'question': e.get(
            'question'), 'answer': file[0]}, json.load(f)))
        print(f'Adding {len(data)} {file[1]} questions...')
        dataset.extend(data)

# with open(f'dataset/question/dataset.json', "w") as f:
#     json.dump(dataset, f, indent=4)

# for file in ['dataset/backup/ran_questions.json', 'dataset/backup/trainset.json', 'dataset/backup/validset.json']:
#     with open(file, "r") as f:
#         dataset.extend(
#             map(lambda e: {'question': e['question'], 'answer': e['answer']}, json.load(f)))
# 
# for file in [(0, 'dataset/backup/yes.txt'), (2, 'dataset/backup/no.txt'), (1, 'dataset/backup/irrelevant.txt')]:
#     with open(file[1], "w") as f:
#         f.write('\n'.join(map(lambda e: e['question'], filter(
#             lambda e: e['answer'] == file[0], dataset))))
