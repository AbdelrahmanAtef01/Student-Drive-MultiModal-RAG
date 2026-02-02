from layout_engine import HybridLayoutEngine
import json
import os

def main():
    engine = HybridLayoutEngine()
    
    # Input PDF
    input_pdf = os.path.join("data_input", "Section 8.docx")
    if not os.path.exists(input_pdf):
        print(f"Error: {input_pdf} not found.")
        return

    # Run Process
    print("Starting Extraction...")
    results = engine.process_file(input_pdf, output_root="data_output", visualize=True)

    # Save Results
    pdf_name = os.path.splitext(os.path.basename(input_pdf))[0]
    output_path = os.path.join("data_output", pdf_name, "results.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Results saved to {output_path}")

if __name__ == "__main__":
    main()