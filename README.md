
## Is normalization needed?
We checked to see if we can feature scale some of the features. As an experiment on the data, I went ahead and tried log normalization on the data. 

First I had to check to see if there were any graphs upon visual inspection of the data where the data starts skewing.

I saw there were some graphs that looked exponential.

I normalized with log, then re-ran another summary. My R^2 jumped from 0.393 to 0.492 and my QQ-Plot looked so pretty with barely any deviation from the line.

Sounds good right? No.

I've concluded that normalization, while it makes things look prettier, can mislead the significance of some of the features and cause biases in the model.  For future work, we can check to see other methods which can show significance between features other than normalization which might obfuscate the results. 


