import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

# ============================================================
# 2. Initialize Ollama
# ============================================================

# Make sure this model is installed:
# ollama pull llama3.2
#
# You can also use:
# qwen2.5:7b
# gemma3:4b
# mistral
# phi4

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")

model = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
)

# ============================================================
# 3. Research Analysis Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template("""
You are an expert academic research assistant.

Your task is to prepare a detailed and well-structured research
analysis for the following research topic:

==================================================
RESEARCH TOPIC:
{topic}
==================================================

Create a comprehensive academic research analysis using the
following structure.

1. RESEARCH TITLE
   - Give a clear, specific, and academically suitable title.

2. ABSTRACT
   - Write approximately 150-250 words.
   - Include the problem, proposed approach, expected contribution,
     and significance.

3. INTRODUCTION
   - Explain the research area.
   - Explain the importance of the topic.
   - Explain its practical and academic relevance.

4. BACKGROUND
   - Explain the concepts, technologies, theories, and domain
     knowledge necessary to understand the topic.

5. RESEARCH PROBLEM
   - Clearly define the research problem.
   - Explain why it requires investigation.
   - Identify important existing challenges.

6. RESEARCH GAP
   - Explain what is missing, insufficient, or unresolved in
     existing approaches.
   - Do not claim that a particular paper has a limitation unless
     the paper/source has actually been provided.

7. AIM OF THE RESEARCH
   - State the main purpose of the study.

8. RESEARCH OBJECTIVES
   - Provide 5-8 specific and measurable objectives.

9. RESEARCH QUESTIONS
   - Provide 5-8 research questions directly related to the objectives.

10. LITERATURE REVIEW THEMES
    - Identify 5-8 major themes.
    - Explain what should be investigated for each theme.
    - Do not invent authors, papers, journals, DOIs, or publication years.

11. PROPOSED METHODOLOGY
    Explain the methodology step by step:

    a. Data collection
    b. Data preprocessing
    c. Exploratory data analysis
    d. Feature engineering
    e. Feature selection
    f. Model development
    g. Model training
    h. Hyperparameter tuning
    i. Model validation
    j. Model comparison
    k. Final evaluation

12. DATASET REQUIREMENTS
    - Explain what data is required.
    - List possible input features.
    - Identify the target variable.
    - Mention possible data sources in general terms.
    - Explain data quality requirements.

13. IMPORTANT VARIABLES / PARAMETERS

    Create a table containing:

    | Variable/Parameter | Description | Data Type |
    | Possible Role | Expected Importance |

14. MACHINE LEARNING / ANALYTICAL METHODS

    Identify appropriate methods for the research topic.

    For each method explain:
    - How it works
    - Why it may be useful
    - Advantages
    - Limitations

15. PERFORMANCE EVALUATION

    Identify appropriate evaluation metrics and explain when
    each metric should be used.

    Classification:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Specificity
    - ROC-AUC
    - Confusion Matrix

    Regression:
    - MAE
    - MSE
    - RMSE
    - R-squared
    - MAPE

16. EXPERIMENTAL DESIGN

    Explain:
    - Training/testing strategy
    - Cross-validation
    - Baseline model
    - Hyperparameter tuning
    - Comparison methodology
    - Reproducibility

17. EXPECTED RESULTS

    - Describe possible types of results.
    - Do NOT invent numerical results.
    - Do NOT invent accuracy, RMSE, F1-score, or other metrics.
    - Clearly label them as expected outcomes.

18. PRACTICAL APPLICATIONS
    - Explain real-world applications.

19. RESEARCH CONTRIBUTION
    - Explain possible academic, technical, and practical contributions.

20. CHALLENGES
    - Identify technical, data-related, and methodological challenges.

21. LIMITATIONS
    - Explain possible limitations of the proposed study.

22. ETHICAL / SECURITY / PRIVACY CONSIDERATIONS
    - Discuss relevant ethical, security, and privacy issues.

23. FUTURE SCOPE
    - Provide 6-10 possible future research directions.

24. CONCLUSION
    - Provide a detailed academic conclusion.

25. IMPLEMENTATION ROADMAP

    Phase 1: Problem definition
    Phase 2: Data collection
    Phase 3: Data preprocessing
    Phase 4: Exploratory analysis
    Phase 5: Model development
    Phase 6: Evaluation
    Phase 7: Deployment
    Phase 8: Documentation

26. SUGGESTED FIGURES AND TABLES

    Suggest useful figures and tables, such as:
    - Data distribution plots
    - Correlation heatmap
    - Feature importance
    - Confusion matrix
    - ROC curve
    - Model comparison
    - Workflow diagram

27. KEYWORDS
    - Provide 8-15 research keywords.

28. SOURCES TO SEARCH FOR

    Suggest reliable academic databases and source types.
    Examples:
    - Google Scholar
    - IEEE Xplore
    - ScienceDirect
    - SpringerLink
    - ACM Digital Library
    - Web of Science
    - Scopus

IMPORTANT ACADEMIC REQUIREMENTS:

- Use formal academic English.
- Use clear headings and subheadings.
- Use tables where appropriate.
- Clearly distinguish facts, assumptions, and expected outcomes.
- Never fabricate statistics.
- Never fabricate experimental results.
- Never fabricate authors.
- Never fabricate citations.
- Never fabricate DOI numbers.
- Never fabricate publications.
- Avoid unsupported claims.
- Do not present generated information as verified literature.
- Make the analysis suitable as a starting point for a research
  paper, project report, thesis, or dissertation.

The final response should be detailed and comprehensive.
""")

# ============================================================
# 4. Get Research Topic
# ============================================================

topic = input("\nEnter your research topic: ").strip()

if not topic:
    raise ValueError("Research topic cannot be empty.")

# ============================================================
# 5. Create Messages
# ============================================================

messages = prompt.format_messages(topic=topic)

# ============================================================
# 6. Generate Research Analysis
# ============================================================

try:

    print("\nConnecting to Ollama...")
    print(f"Model: {MODEL_NAME}")

    result = model.invoke(messages)

    print("\n")
    print("=" * 80)
    print("                    RESEARCH ANALYSIS")
    print("=" * 80)

    print(f"\nTopic: {topic}\n")

    print(result.content)

    print("\n")
    print("=" * 80)
    print("                  END OF ANALYSIS")
    print("=" * 80)

except Exception as e:

    print("\n" + "=" * 80)
    print("ERROR WHILE GENERATING RESEARCH ANALYSIS")
    print("=" * 80)

    print(f"\nError Type : {type(e).__name__}")
    print(f"Error      : {str(e)}")

    print("\nPossible solutions:")
    print("1. Make sure Ollama is running.")
    print("2. Check whether the model is installed.")
    print(f"3. Run: ollama pull {MODEL_NAME}")
    print("4. Check: ollama list")

