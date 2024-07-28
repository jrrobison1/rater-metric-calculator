
from langchain_core.prompts import PromptTemplate

basic_prompt = PromptTemplate.from_template(
    template="""
I performed an AESTHEMOS 1-5 scale "survey" to 7 LLMs and 1 Human, reviewing 150 word snippets of a book, and asked the question "How strongly did you experience a sense of beauty or being moved by this passage?" Survey ratings corresponded to 1=none, 2=some, 3=moderately, 4=strongly, 5=very strongly. Following is the result rater agreement analysis. Interpret the analysis in the following format: 
## Overall Interpretation 
## Rating Distribution 
## Pairwise 
### Pairwise general interpretation 
### Pairwise standouts (particularly interesting agreement pairs 
### Clusters. (Are there any "clusters" of LLMs that all agree with each other more?) 
## Conclusion 
### Overall insights (Think very carefully and including 3 insights) 
### Suggested research directions (Max of 3) 

Here are the results for analysis:

-----
{input}
-----
"""
)
