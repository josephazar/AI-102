import streamlit as st
import os
from content_understanding import process_video, generate_summary, generate_report, chat_with_video
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from azure.core.credentials import AzureKeyCredential
from PIL import Image
import base64

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants from environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_CU_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CU_EMBEDDING_DEPLOYMENT_NAME")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_SERVICE_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_CU_INDEX_NAME")
AZURE_AI_CU_KEY = os.getenv("AZURE_AI_CU_KEY")

def show_content_understanding():
    st.title("Azure Content Understanding for Real Estate")
    st.write("This demo showcases how Azure Content Understanding can process a pre-indexed video to generate a real estate listing, provide a summary, and enable chatting with the video content.")

    # Use pre-existing video
    video_file = os.path.join("video", "sample_video.mp4")  # Path to pre-indexed video in DOCUMENTS_DIR

    if not os.path.exists(video_file):
        st.error(f"Video file {video_file} not found. Please ensure the video is available in the 'video' directory.")
        return

    # Initialize vector store (assumes video is already processed and indexed)
    aoai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        openai_api_version="2025-01-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_AI_CU_KEY
    )
    vector_store = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=aoai_embeddings.embed_query
    )

    # Display key frames
    st.subheader("Key Frames")
    keyframe_files = [os.path.join("frames", f) for f in os.listdir("frames") if f.endswith(".jpg")]
    if not keyframe_files:
        st.warning("No key frames found in the 'frames' directory.")
    else:
        for img_file in keyframe_files[:5]:  # Show first 5
            st.image(img_file, caption=os.path.basename(img_file), use_column_width=True)

    # Show video summary
    st.subheader("Video Summary")
    # For demo, process video to get summary; in practice, this could be cached or precomputed
    vector_store, video_result, keyframe_files = process_video(video_file)
    summary = generate_summary(video_result)
    st.write(summary)

    # Chat with video
    st.subheader("Chat with Your Video")
    query = st.text_input("Ask a question about the video content:")
    if query:
        with st.spinner("Searching..."):
            results = chat_with_video(query, vector_store)
            if not results:
                st.write("No relevant segments found for your query.")
            else:
                for i, result in enumerate(results):
                    st.write(f"**Result {i+1}:**")
                    st.write(f"- **Scene Description**: {result['scene_description']}")
                    st.write(f"- **Kind**: {result['kind']}")
                    st.write(f"- **Start Time**: {result['start_time_ms']} ms")
                    st.write(f"- **End Time**: {result['end_time_ms']} ms")
                    st.write(f"- **Transcript Phrases**: {', '.join(result['transcript_phrases'])}")

    # Generate report
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            report_file = generate_report(video_result, keyframe_files, summary)
            with open(report_file, "rb") as file:
                st.download_button(
                    label="Download Report",
                    data=file,
                    file_name="video_analysis_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

if __name__ == "__main__":
    show_content_understanding()