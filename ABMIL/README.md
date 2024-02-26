
## AB_MIL

AB_MIL(AtAttention-Based Multiple Instance Learning (AB_MIL) is an advanced form of conventional multi-instance learning (MIL) methods.

Existing MIL methods have the disadvantage of assuming that all instances contribute equally to the label.

However, in practice, some instances may have a greater impact on label decisions.

Attention-based MIL introduces an attention mechanism to address these issues. Therefore, it is believed that more accurate prediction is possible by highlighting important instances through the attention mechanism.

For each instance, the embedding Z, multiplied by the attention, is expressed as follows.

$$
z = \sum_{k=1}^K a_k h_k
$$

Here $a_k$ stands for the attention of the $kth$ instance and $h_k$ stands for the embedding of the instance. The attention expression is as follows.


$$
a_k=\frac{\exp{w^{\top} \tanh V h_k^{\top}}}{\sum_{j} \exp w^{\top} \tanh V h_j^{\top}}
$$

As above, the result of introducing attention and learning well is expressed as below. Each instance is expressed as a small patch, and when the patch is multiplied by the attention, it is expressed as d. The more pronounced it is, the higher the attention is (patch).

>**Problem**
However, when we extract instance embeddings through Resnet during learning and check the attention, we observe that the attention of the instance is spread evenly and cannot concentrate on one.

>**Solutions**
It was concluded that the pretrained model should be fine tuned as a solution. Therefore, Binary classification was set as objective function for recurrence for each instance and the backbone was finetuned, and when the finetuned backbone and AB-MIL were applied, I observed that it was well-attentioned. In other words, it was confirmed that 90% attention was applied to the top 50% of instances.

Results: Due to lack of time, I used only 1/5 of the entire dataset, and when I submitted it by learning as above, I found 0.42. If I had learned it with the entire dataset and appropriate normalization and augmentation, wouldn't it have resulted in better results.



<!--

AB_MIL(Attention-Based Multiple Instance Learning)은 기존 다중 인스턴스 학습 (MIL) 방법의 한 발전된 형태입니다. 

기존 MIL 방법은 모든 인스턴스가 레이블에 동일하게 기여한다고 가정하는 단점이 있습니다. 

하지만 실제로는 일부 인스턴스가 레이블 결정에 더 큰 영향을 미칠 수 있습니다. 

어텐션 기반 MIL은 이러한 문제를 해결하기 위해 어텐션 메커니즘을 도입합니다. 따라서 어텐션 메커니즘을 통해 중요한 인스턴스를 강조함으로 더 정확한 예측이 가능할것으로 판단됩니다. 

각 인스턴스마다 어텐션을 곱해서 합한 임베딩 Z는 다음과 같이 나타냅니다.


$$
z = \sum_{k=1}^K a_k h_k
$$

여기서 $a_k$는 $k번째$ 인스턴스의 어텐션을 의미하고 $h_k$는 인스턴스의 임베딩을 의미합니다. 어텐션의 식은 다음과 같습니다.

$$
a_k=\frac{\exp{w^{\top} \tanh V h_k^{\top}}}{\sum_{j=1}^K \exp w^{\top} \tanh V h_j^{\top}}
$$

위처럼 어텐션을 도입해서 잘 학습이 된결과는 아래처럼 표현 됩니다. 각 하나하나의 인스턴스들은 작은 패치로 표현이 되고, 패치에 어탠션을 곱하면 d처럼 나타내집니다. 뚜렷할수록 어텐션이 높은 인스턴스(패치)입니다.

**문제** 
하지만, 학습시에 Resnet을 통해 인스턴스 임베딩을 뽑고, 어텐션을 확인해본결과 인스턴스의 어텐션이 골고루 퍼져서 하나에 집중 못하는것으로 관측되었습니다.

**솔루션** 
솔루션으로는 pretrained 모델을 파인튜닝 시켜야한다고 결론에 이르렀습니다. 따라서 인스턴스마다 재발여부에 Binary classification을 objective function으로 놓고 backbone을 finetuning을 시켰고, 이렇게 finetuning 시킨 backbone과 AB-MIL을 적용시켰더니 잘 어텐션 되는 모습을 관측하였습니다. 즉, 상위 50프로의 인스턴스에 90프로 어텐션이 적용됨을 확인하였습니다.

결과 : 시간부족으로 전체 데이터셋의 1/5만 사용해서, 위 처럼 학습을 시켜 제출해보니 0.42가 나왔습니다.  전체 데이터셋과 적절한 normalization, augmentation으로 학습시켰다면, 더 좋은 결과에 이르지 않았을까 싶습니다.
-->
