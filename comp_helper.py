from numpy import log2, prod, mean



# supplymnet functions for entropy calculation
def shannon_entropy(problist):
    """Shannon Entropy: negative sum over all probabilities*log2_probabilities
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    return -1 * sum([prob * log2(prob) for prob in problist])

def nltk_entropy(problist):
    """From nltk.lm: negative average log2_probabilities
    https://www.nltk.org/api/nltk.lm.api.html#nltk.lm.api.LanguageModel.entropy
    """
    return -1 * mean([log2(prob) for prob in problist])

def perplexity(entropy):
    """From nltk.lm: 2**entropy
    This is also a general formula for entropy, not only by NLTK
    https://www.nltk.org/api/nltk.lm.api.html#nltk.lm.api.LanguageModel.perplexity
    """
    return pow(2.0, entropy)

def jurafsky_perplexity(problist):
    """From Jurafsky, Stanford NLP: product of all probabilities** -1/count_of_probabilities
    https://www.youtube.com/watch?v=NCyCkgMLRiY
    """
    probs = [prob for prob in problist]
    return pow(prod(probs), -1/len(probs))

def uncertainty(problist):
    prob_sum =  sum(problist) # just to be sure our total probability mass == 1
    shan_ent = shannon_entropy(problist)
    shan_prplx = perplexity(shan_ent)
    nltk_ent = nltk_entropy(problist)
    nltk_prplx = perplexity(nltk_ent)
    juraf_prplx = jurafsky_perplexity(problist)
    
    # print(
    #     f"Probability sum: {prob_sum}\n\n"
    #     f"Shannon entropy: {shan_ent}\n"
    #     f"Shannon perplexity: {shan_prplx}\n\n"
    #     f"NLTK entropy: {nltk_ent}\n"
    #     f"NLTK perplexity: {nltk_prplx}\n\n"
    #     f"Jurafsky perplexity: {juraf_prplx}")
    return [prob_sum, shan_ent, shan_prplx, nltk_ent, nltk_prplx, juraf_prplx]