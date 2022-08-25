import torch

from transformers import BertTokenizer
from bert_classifier import BertClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "./saved_model"
model = BertClassifier.from_pretrained(model_path)
model.to(device)
model.eval()
# print(model)

tokenizer = BertTokenizer.from_pretrained(model_path)

example_text = 'spirit awards hail sideways the comedy sideways has dominated this year s independent spirit awards  winning all six of the awards for which it was nominated.  it was named best film while alexander payne won best director and best screenplay  along with writing partner jim taylor. it also won acting awards for stars paul giamatti  thomas haden church and virginia madsen. sideways is tipped to do well at sunday s oscars  with five nominations.  the awards  now in their 20th year  are given to films made outside the traditional studio system  and are traditionally held the day before the oscars. other winners included catalina sandino moreno  who took best actress for her role as a drug smuggler in the colombian drama maria full of grace. moreno is also nominated for best actress at the oscars. the best first screenplay award went to joshua marston for maria full of grace. scrubs star zach braff won the award for best first feature for garden state  which he wrote  directed and starred in. oscar-nominated euthanasia film the sea inside from spain won best foreign film  while metallica: some kind of monster was awarded best documentary. actor rodrigo de la serna took the best debut performance prize for the motorcycle diaries. the awards are voted for by the 9 000 members of the independent feature project/los angeles  which includes actors  directors  writers and other industry professionals. last year s big winner  lost in translation  went on to win the oscar for best original screenplay  for writer-director sofia coppola.'

input = tokenizer(example_text,padding='max_length', max_length = 250, 
                       truncation=True, return_tensors="pt")

mask = input['attention_mask'].to(device)
input_id = input['input_ids'].squeeze(1).to(device)
with torch.no_grad():
    output = model(input_id, mask)

print(output)

