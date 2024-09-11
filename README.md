***Use Random forest (RF) to estimate maximum ROI solution on media investment***

A previous requirement involved always-on conversion advertising projects where clients often asked, "How should I allocate my daily budget? How long should a set of creatives run on a specific ad platform?" In the past, without detailed calculations, the operations team would provide a rather vague experiential value (such as ￥800,000 per day, and a set of creatives running for 14 days). However, after accumulating a certain amount of data, we could attempt to use algorithms to complete this prediction (or summarization), thus providing more precise answers.

之前遇到的一个需求是一些始终在线的转化类广告项目中，客户经常会问：“我应该如何分配每日的预算？在某个广告平台上，一套素材应该投放多少天？”过去，运营团队在没有细致计算的情况下，可能会给出一个较为模糊的经验值建议（例如每天80万元，一套素材投放14天）。然而，在积累了足够的数据之后，我们可以通过算法来进行这种预测（或者说总结），从而提供更加准确的答案。

For this task, Gaussian Process Regression and Polynomial Regression (up to cubic terms) were tried, but in practice, it was found that the Random Forest algorithm performed better. Therefore, it's currently recommended to use the Random Forest algorithm to complete the prediction.

这个任务之前尝试过使用高斯过程回归和多项式回归（最高扩展到三次项），但是实践中发现随机森林算法的效果更好，因此目前主要推荐使用随机森林算法来完成预测。

The script was write to run on Google Colab. For those who need to run it locally, adjust the file reading and storage path modules accordingly.

这个脚本是在Google Colab上运行的，如果需要在本地运行，只需要调整文件读取和存储的路径即可。

**Usage** 

You can directly copy the code from and run [Run.py](https://github.com/Chaoshcx/roi-estimate/blob/main/Run.py) on Google Colab, or adjust the doc placement code part and run on your own computer

Colab用户直接复制Run.py的代码执行即可（超参数调整在代码里有说明）

**Process**

![](https://github.com/Chaoshcx/roi-estimate/blob/main/process.png)

**Examples**
RF model predict

![](https://github.com/Chaoshcx/roi-estimate/blob/main/examples/Model%20performance%20RF.png)

Polynomial model predict

![](https://github.com/Chaoshcx/roi-estimate/blob/main/examples/Model%20performance%20Polynomial%20.png)

ROI trend based on certain budget

![](https://github.com/Chaoshcx/roi-estimate/blob/main/examples/ROI%20trend1.png)

ROI trend based on certain days
![](https://github.com/Chaoshcx/roi-estimate/blob/main/examples/ROI%20trend2.png)

