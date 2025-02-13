import os

train_path ="./MSRA/train1.txt"
test_path = "./MSRA/testright1.txt"

train_bmes = "./MSRA/train_dev.char.bmes"
test_bmes = "./MSRA/test.char.bmes"

ftrain = open(train_path,"r",encoding="utf-8")
fbmes = open(train_bmes,"w",encoding="utf-8")

ftest = open(test_path,"r",encoding="utf-8")
testbmes = open(test_bmes,"w",encoding="utf-8")


dict = {"NR":"NAME","NT":"ORG","NS":"CONT"}

lines = ftrain.readlines()
for line in lines:
    line = line.strip()
    words = line.split()
    for traindata in words:
        char_lable = traindata.split("/")
        index = 0

        for i in char_lable[0]:
            if i in ['。','！','？'] and char_lable[1].lower() == "o":
                fbmes.write(i + ' ' + char_lable[1].upper() + '\n')

            elif char_lable[1] == "o":

                fbmes.write(i +" "+ char_lable[1].upper() + '\n')


            else:
                if len(char_lable[0])==1:
                    fbmes.write(i + ' S-' + char_lable[1].upper() + '\n')
                elif i == char_lable[0][0] and index == 0:
                    fbmes.write(i+' B-' + char_lable[1].upper()+'\n')
                    index +=1

                elif i == char_lable[0][-1] and index != len(char_lable[0]):
                    fbmes.write(i+' E-' + char_lable[1].upper()+'\n')
                    index += 1

                else:
                    fbmes.write(i+' M-' + char_lable[1].upper()+'\n')
                    index += 1


    fbmes.write('\n')


lines = ftest.readlines()
for line in lines:
    line = line.strip()
    words = line.split()
    for traindata in words:
        char_lable = traindata.split("/")
        index = 0
        for i in char_lable[0]:

            if i in ['。', '！', '？'] and char_lable[1].lower() == "o":
                testbmes.write(i + ' ' + char_lable[1].upper() + '\n')

            elif char_lable[1] == "o":

                testbmes.write(i +" "+ char_lable[1].upper() + '\n')

            else:
                if len(char_lable[0]) == 1:
                    testbmes.write(i + ' S-' + char_lable[1].upper() + '\n')
                else:
                    if i == char_lable[0][0] and index == 0:
                        testbmes.write(i + ' B-' + char_lable[1].upper() + '\n')
                        index += 1

                    elif i == char_lable[0][-1] and index == len(char_lable[0]) - 1:
                        testbmes.write(i + ' E-' + char_lable[1].upper() + '\n')
                        index += 1
                    else:
                        testbmes.write(i + ' M-' + char_lable[1].upper() + '\n')
                        index += 1


    testbmes.write('\n')

ftrain.close()
fbmes.close()
ftest.close()
testbmes.close()

def create_cliped_file(fp):
    f = open(fp,'r',encoding='utf-8')
    fp_out = fp + '_clip2'
    f_out = open(fp_out,'w',encoding='utf-8')
    now_example_len = 0
    # cliped_corpus = [[]]
    # now_example = cliped_corpus[0]

    lines = f.readlines()
    last_line_split = ['','']
    for line in lines:
        line_split = line.strip().split()

        print(line,end='',file=f_out)
        now_example_len += 1
        if len(line_split) == 0 or \
                (line_split[0] in ['。','！','？']
                 and line_split[1] == 'O' and now_example_len>170):
            print('',file=f_out)
            now_example_len = 0
        elif ((line_split[0] in [',',"、","，",'；'] or (now_example_len>1 and last_line_split[0] == '…' and line_split[0] == '…'))
                 and line_split[1] == 'O' and now_example_len>170):
            print('',file=f_out)
            now_example_len = 0

        elif line_split[1][0].lower() == 'e' and now_example_len>170:
            print('',file=f_out)
            now_example_len = 0

        last_line_split = line_split

    f_out.close()
    f_check = open(fp_out,'r',encoding='utf-8')
    lines = f_check.readlines()
    cliped_examples = [[]]
    now_example = cliped_examples[0]
    for line in lines:
        line_split = line.strip().split()
        if len(line_split) == 0:
            cliped_examples.append([])
            now_example = cliped_examples[-1]
        else:
            now_example.append(line.strip())

    check = 0
    max_length = 0
    for example in cliped_examples:
        if len(example)>170:
            print(len(example),''.join(map(lambda x:x.split(' ')[0],example)))
            check = 1

        max_length = max(max_length,len(example))

    print('最长的句子有:{}'.format(max_length))

    if check == 0:
        print('没句子超过170的长度')


create_cliped_file('./MSRA/train_dev.char.bmes')
create_cliped_file('./MSRA/test.char.bmes')