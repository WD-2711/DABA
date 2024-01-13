# DABA
Reproduction of Opportunistic Backdoor Attacks's paper, thanks to the students at the National University of Defense Technology for providing the data set for the reproduction of this paper.

The author has uploaded the relevant code to https://github.com/lqsunshine/DABA. I just tidied up the code a little bit.

1. s1_train_test.py. Generate clean speech classification models.
2. lib_trigger_injection.py. According to the data_test and data_train folders, poisoning samples are generated and stored in the data_test_poison and data_train_poison folders.
3. s2_train_test_on_poison.py. Based on the poisoning samples, a poisoning model is generated.
3. data_train and data_test can download from author's github link.

The call graph of the main files is as follows:

![s1_train_test](s1_train_test.png)

![s1_train_test](lib_trigger_injection.png)
