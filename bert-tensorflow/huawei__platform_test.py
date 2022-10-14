import moxing as mox
import tensorflow as tf
import moxing.tensorflow as mtf
import os
mtf.cache()
print(help(mox))
print(mox.file.exists('obs://tf-bert-base/huawei/stopwords.txt'))
print(os.path.exists('obs://tf-bert-base/huawei/stopwords.txt'))

mox.file.copy('obs://tf-bert-base/huawei/stopwords.txt', '/home/work/user-job-dir/stopwords.txt')

print(mox.file.exists('/home/work/user-job-dir/stopwords.txt'))
print(os.path.exists('/home/work/user-job-dir/stopwords.txt'))