from workflow import app_workflow
from chat_graph import chat_app

# Generate the Mermaid diagram binary
png_data = app_workflow.get_graph().draw_mermaid_png()

# Save to a file
output_file = "rag_pipeline_architecture.png"
with open(output_file, "wb") as f:
    f.write(png_data)

print(f"Graph visualization saved to: {output_file}")

# Generate the Mermaid diagram binary
png_data = chat_app.get_graph().draw_mermaid_png()

# Save to a file
output_file = "chat_pipeline_architecture.png"
with open(output_file, "wb") as f:
    f.write(png_data)

print(f"Graph visualization saved to: {output_file}")