Random forest algorithm to estimate maximum ROI solution on media investment

A previous requirement involved always-on conversion advertising projects where clients often asked, "How should I allocate my daily budget? How long should a set of creatives run on a specific ad platform?" In the past, without detailed calculations, the operations team would provide a rather vague experiential value (such as ￥800,000 per day, and a set of creatives running for 14 days). However, after accumulating a certain amount of data, we could attempt to use algorithms to complete this prediction (or summarization), thus providing more precise answers.

For this task, Gaussian Process Regression and Polynomial Regression (up to cubic terms) were tried, but in practice, it was found that the Random Forest algorithm performed better. Therefore, it's currently recommended to use the Random Forest algorithm to complete the prediction.

The script was write to run on Google Colab. For those who need to run it locally, adjust the file reading and storage path modules accordingly.

**Usage** 

You can directly copy the code from and run [Run.py](https://github.com/Chaoshcx/roi-estimate/blob/main/Run.py) on Google Colab, or adjust the doc placement code part and run on your own computer

**Process**

![](https://github.com/Chaoshcx/roi-estimate/blob/main/process.png)

**Examples**
