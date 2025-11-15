from .pipeline import SimplePromptPipeline

context = """<CONTEXT>"""

question = "<QUESTION>"

pipeline = SimplePromptPipeline()
result = pipeline.run(context, question)

print("\nAnswer:", result["answer"])
