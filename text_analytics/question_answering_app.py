import streamlit as st
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering import QuestionAnsweringClient

def show_question_answering():
    # Initialize the Azure Question Answering client
    endpoint = os.getenv("LANGUAGE_SERVICE_ENDPOINT")
    credential = AzureKeyCredential(os.getenv("LANGUAGE_SERVICE_KEY"))
    if not endpoint or not credential:
        st.error("Please set LANGUAGE_SERVICE_ENDPOINT and LANGUAGE_SERVICE_KEY in your .env file.")
        return
    qa_client = QuestionAnsweringClient(endpoint, credential)

    # Define interesting pre-loaded examples
    examples = {
        "Alice in Wonderland": {
            "text": (
                "Alice was beginning to get very tired of sitting by her sister on the bank, "
                "and of having nothing to do: once or twice she had peeped into the book her sister "
                "was reading, but it had no pictures or conversations in it, 'and what is the use of "
                "a book,' thought Alice 'without pictures or conversations?'"
            ),
            "question": "Why was Alice getting tired?"
        },
        "Water Cycle": {
            "text": (
                "The water cycle is the continuous process of water evaporating, condensing, and "
                "precipitating. It starts with the sun heating up water in rivers, lakes, and oceans, "
                "causing it to evaporate into water vapor. This vapor rises into the atmosphere, "
                "where it cools and condenses to form clouds. Eventually, the water falls back to "
                "the earth as precipitation, such as rain or snow."
            ),
            "question": "What causes water to evaporate in the water cycle?"
        },
        "Moon Landing": {
            "text": (
                "On July 20, 1969, NASA's Apollo 11 mission successfully landed the first humans on "
                "the Moon. Astronauts Neil Armstrong and Buzz Aldrin stepped onto the lunar surface, "
                "while Michael Collins orbited above in the command module. Armstrong famously said, "
                "'That's one small step for man, one giant leap for mankind.' The mission was a "
                "significant achievement in the space race between the United States and the Soviet Union."
            ),
            "question": "Who were the first two people to walk on the Moon?"
        },
        "History of the Internet": {
            "text": (
                "The internet began as a project called ARPANET, funded by the U.S. Department of "
                "Defense in the 1960s. It was designed to allow communication between different "
                "computer networks. In 1983, ARPANET adopted the TCP/IP protocol, which became the "
                "foundation of the modern internet. The World Wide Web, invented by Tim Berners-Lee "
                "in 1989, made the internet more accessible to the public by introducing hyperlinks "
                "and web browsers."
            ),
            "question": "What protocol did ARPANET adopt in 1983?"
        }
    }

    # Sidebar for example selection
    st.sidebar.subheader("Explore Examples")
    example_options = ["Custom"] + list(examples.keys())
    selected_example = st.sidebar.selectbox(
        "Select an example or choose 'Custom' to enter your own text",
        example_options
    )

    # Populate session state when an example is selected (only if it changes)
    if selected_example != "Custom":
        st.session_state["qa_text"] = examples[selected_example]["text"]
        st.session_state["qa_question"] = examples[selected_example]["question"]
    elif selected_example == "Custom" and "qa_text" not in st.session_state:
        # Initialize with empty values for Custom on first load
        st.session_state["qa_text"] = ""
        st.session_state["qa_question"] = ""

    # Main interface
    st.title("Azure Question Answering Explorer")
    st.markdown(
        """
        Ask questions about text documents using Azure's AI-powered Question Answering service. 
        Select an example from the sidebar or input your own text and question below!
        """
    )

    # Input fields (let widgets manage their own state)
    text = st.text_area(
        "Enter or edit the text document",
        value=st.session_state.get("qa_text", ""),
        height=200,
        key="qa_text",
        help="Paste your text here (English recommended)."
    )
    question = st.text_input(
        "Enter your question",
        value=st.session_state.get("qa_question", ""),
        key="qa_question",
        help="Ask a question about the text above."
    )

    # Button to get the answer
    if st.button("Get Answer"):
        if not text.strip() or not question.strip():
            st.warning("Please provide both text and a question.")
        else:
            with st.spinner("Querying Azure AI..."):
                try:
                    response = qa_client.get_answers_from_text(
                        question=question,
                        text_documents=[{"id": "1", "text": text}]
                    )
                    if response.answers:
                        top_answer = response.answers[0]
                        st.success(f"**Answer:** {top_answer.answer}")
                        st.info(f"**Confidence:** {top_answer.confidence:.2f}")
                    else:
                        st.warning("No answer found in the text.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")



if __name__ == "__main__":
    show_question_answering()