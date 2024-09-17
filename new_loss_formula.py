delLi = Li(t+1) - Li(t)
delWi = Wi(t+1) - Wi(t)
deltaWi = delLi / delWi
Wi(t+1) = Wi(t) - lr(delWi)
# Sum of all Wi must be 1
# 5 Losses means i ranges from 0 to 4. Each Wi corresponds to a particular loss
# t is the epoch number
total_loss = Sum(Wi * Li)

I have a proposal for a new loss formula. Wi corresponds to the weight of each loss.
Tell me what you understand from this formula.