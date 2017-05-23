#Answer Selection in IRGAN

## how to run
prepare the test dataset [InsuranceQA](https://github.com/codekansas/insurance_qa_python.git)
<pre><code>python dataPrepare.py</code></pre>
run the baseline
<pre><code>python baseline.py</code></pre>
run in IRGAN framework
<pre><code>python irgan.py</code></pre>


## note


The main reason to choose InsuanceQA as our benchmark due to that there is a bigger size of negative answer pool. Meanwhile, it is relatively large to alleviate overfitting for training a large neural network. In this test setting, we will sample 499 negative answers for each question-answer pair. (A quesiton with multiple answers will be composed as many question-answer pairs).  

There are enough unlabeled data  in  websearch (semi version) and item recommendation (two other task in this IRGAN paper), which is essential for generator to sample more competitive negative answers and thus yield a signigicant improvement in performance. In the InsuranceQA dataset,  However, the sampled negative answers are usually too weak, and the positive answers in a single question are usually sparse, thus the generator hardly performes well.



if you want to only improve the performance of InsuranceQA, you should
1. use a well-trained word embedding instead of a random one.
2. add some complex mechanisms, like attentive pooling.
3. build more competitive negative answers pools, may derived from external corpus. 

## contact
mail to waby@tju.edu.cn