from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os

# ---------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------
load_dotenv()
#
#api_key = os.getenv("GOOGLE_API_KEY")
#
#if not api_key:
#    raise ValueError(
#        "GOOGLE_API_KEY not found. "
#        "Please add GOOGLE_API_KEY=your_api_key to your .env file."
#    )
#
# ---------------------------------------------------
# 2. Initialize Ollama
# ---------------------------------------------------
model = ChatOllama(
    temperature=0.2,
)

# ---------------------------------------------------
# 3. Research analysis prompt
# ---------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are an expert academic research assistant.

Your task is to prepare a detailed and well-structured research analysis
for the following research topic:

==================================================
RESEARCH TOPIC:
{topic}
==================================================

Create a comprehensive academic research analysis using the following
structure:

1. RESEARCH TITLE
   - Give a clear, specific and academically suitable title.

2. ABSTRACT
   - Write a detailed abstract of approximately 150-250 words.
   - Include the problem, proposed approach, expected contribution,
     and significance.

3. INTRODUCTION
   - Explain the research area.
   - Introduce the importance of the topic.
   - Explain the practical and academic relevance.

4. BACKGROUND
   - Explain the concepts, technologies, theories, or domain knowledge
     necessary to understand the topic.

5. RESEARCH PROBLEM
   - Clearly define the problem.
   - Explain why the problem requires investigation.
   - Identify important existing challenges.

6. RESEARCH GAP
   - Explain what is missing, insufficient, or unresolved in existing
     approaches.
   - Do not claim that a specific paper has a particular limitation
     unless the source is actually provided.

7. AIM OF THE RESEARCH
   - State the main purpose of the study.

8. RESEARCH OBJECTIVES
   - Provide 5-8 specific and measurable objectives.

9. RESEARCH QUESTIONS
   - Provide 5-8 research questions directly related to the objectives.

10. LITERATURE REVIEW THEMES
    - Identify 5-8 major themes that should be covered in the literature.
    - For each theme, explain what should be investigated.
    - Do not invent authors, papers, journals, DOIs, or publication years.

11. PROPOSED METHODOLOGY
    Explain the complete research methodology step by step:
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
    - Variable/parameter name
    - Description
    - Data type
    - Possible role
    - Expected importance

14. MACHINE LEARNING / ANALYTICAL METHODS
    Identify appropriate methods for the research topic.
    For each method explain:
    - How it works
    - Why it may be useful
    - Advantages
    - Limitations

15. PERFORMANCE EVALUATION
    Identify appropriate evaluation metrics.
    Explain when each metric should be used.

    For classification, consider:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Specificity
    - ROC-AUC
    - Confusion Matrix

    For regression, consider:
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
    - Describe what types of results the research may produce.
    - Do NOT invent numerical accuracy, RMSE, F1, or other results.
    - Clearly label these as expected outcomes.

18. PRACTICAL APPLICATIONS
    - Explain how the research could be applied in real-world situations.

19. RESEARCH CONTRIBUTION
    - Explain possible academic, technical and practical contributions.

20. CHALLENGES
    - Identify likely technical, data-related and methodological challenges.

21. LIMITATIONS
    - Explain possible limitations of the proposed study.

22. ETHICAL / SECURITY / PRIVACY CONSIDERATIONS
    - Mention relevant issues where applicable.

23. FUTURE SCOPE
    - Provide 6-10 possible future research directions.

24. CONCLUSION
    - Provide a detailed academic conclusion summarizing the research.

25. IMPLEMENTATION ROADMAP
    Provide a step-by-step development plan:
    Phase 1: Problem definition
    Phase 2: Data collection
    Phase 3: Data preprocessing
    Phase 4: Exploratory analysis
    Phase 5: Model development
    Phase 6: Evaluation
    Phase 7: Deployment
    Phase 8: Documentation

26. SUGGESTED FIGURES AND TABLES
    Suggest useful figures and tables for the research paper, such as:
    - Data distribution plots
    - Correlation heatmap
    - Feature importance
    - Confusion matrix
    - ROC curve
    - Model comparison
    - Workflow diagram

27. KEYWORDS
    - Give 8-15 research keywords.

28. SOURCES TO SEARCH FOR
    - Suggest types of reliable sources and databases to consult.
    - Do not fabricate references.
    - Examples may include Google Scholar, IEEE Xplore,
      ScienceDirect, SpringerLink, ACM Digital Library, etc.

ACADEMIC REQUIREMENTS:
- Use formal academic English.
- Use headings and subheadings.
- Provide detailed explanations.
- Use tables where appropriate.
- Clearly distinguish known information, assumptions, and expected outcomes.
- Never fabricate statistics, experimental results, authors, citations,
  DOI numbers, or publications.
- Avoid unsupported claims.
- Make the analysis suitable as a starting point for a research paper,
  project report, thesis, or dissertation.

The final response should be detailed and comprehensive.
""")

# ---------------------------------------------------
# 4. Get research topic from the user
# ---------------------------------------------------
topic = input("\nEnter your research topic: ").strip()

if not topic:
    raise ValueError("Research topic cannot be empty.")

# ---------------------------------------------------
# 5. Format prompt
# ---------------------------------------------------
messages = prompt.format_messages(topic=topic)

# ---------------------------------------------------
# 6. Generate research analysis
# ---------------------------------------------------
try:
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
    print("\nError while generating research analysis:")
    print(type(e).__name__)
    print(str(e))