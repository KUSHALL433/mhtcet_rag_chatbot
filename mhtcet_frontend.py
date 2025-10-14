## NEW CODE ##
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from mhtcet_database_backend import rag_agent  # your backend agent

# ---- PAGE CONFIG ----
st.set_page_config(page_title="MHT-CET RAG Chatbot", layout="centered")

st.title("🎓 MHT-CET Exam RAG Chatbot")

# ---- SESSION STATE INITIALIZATION ----
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())  # unique per user
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

config = {'configurable': {'thread_id': st.session_state['session_id']}}

# ---- CLEAR CHAT BUTTON ----
if st.button("🗑️ Clear Chat"):
    st.session_state['message_history'] = []

# ---- DISPLAY EXISTING MESSAGES ----
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# ---- USER INPUT ----
user_input = st.chat_input("Ask your MHT CET related question...")

if user_input:
    # Show user message immediately
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # ---- CALL RAG AGENT ----
    try:
        response = rag_agent.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)
        ai_message_content = response['messages'][-1].content
    except Exception as e:
        ai_message_content = f"⚠️ I ran into some errors. This is a personal project... Probably Token limits reached.... Come back tomorrow."

    # Show assistant response
    with st.chat_message('assistant'):
        st.markdown(ai_message_content)

    # Save to chat history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message_content})


import atexit
from mhtcet_database_backend import delete_user_session

# Delete user data when session ends
def cleanup_session():
    delete_user_session(st.session_state['session_id'])

atexit.register(cleanup_session)

## END ##

## *************************OLD CODE ***************************** ##

# import streamlit as st
# import uuid
# from mhtcet_database_backend import rag_agent,retrieve_all_threads
# from langchain_core.messages import HumanMessage,AIMessage

# def generate_thread_id():
#     thread_id=uuid.uuid4()
#     return thread_id

# def reset_chat():
#     new_thread_id = generate_thread_id()
#     st.session_state['thread_id'] = new_thread_id
#     st.session_state['message_history'] = []
#     add_thread(new_thread_id)

# def add_thread(thread_id):
#     if thread_id not in st.session_state['chat_threads']:
#         st.session_state['chat_threads'].append(thread_id)

# def load_conversation(thread_id):
#     # return rag_agent.get_state(config={'configurable':{'thread_id':thread_id}}).values['messages']
#     state = rag_agent.get_state(config={'configurable': {'thread_id': thread_id}})
#     if state and 'messages' in state.values:
#         return state.values['messages']
#     else:
#         return []


# if 'message_history' not in st.session_state:
#     st.session_state['message_history']=[]
# if 'thread_id' not in st.session_state:
#     st.session_state['thread_id']=generate_thread_id()
# if 'chat_threads' not in st.session_state:
#     st.session_state['chat_threads']=retrieve_all_threads()


# add_thread(st.session_state['thread_id'])

# st.sidebar.title("MHTCET Exam RAG ChatBot.")

# if st.sidebar.button("New Chat"):
#     reset_chat()

# st.sidebar.header('My Chats')

# for thread_id in st.session_state['chat_threads'][::-1]:
#     if st.sidebar.button(str(thread_id)):
#         st.session_state['thread_id']=thread_id
#         messages=load_conversation(thread_id)
#         temp_messages=[]
#         for message in messages:
#             if isinstance(message,HumanMessage):
#                 role='user'
#                 temp_messages.append({'role':role,'content':message.content})
#             elif isinstance(message, AIMessage):
#                 if message.content:
#                     role='assistant'
#                     temp_messages.append({'role':role,'content':message.content})
#         st.session_state['message_history']=temp_messages
        

# thread_id=st.session_state['thread_id']
# config={'configurable':{'thread_id':thread_id}}
# messages = st.session_state['message_history']

# # ✅ Prevent crash when opening new empty chat
# if len(messages) == 0:
#     st.info("💬 Start the conversation by asking your first question!")
# else:
#     for message in messages:
#         with st.chat_message(message['role']):
#             st.text(message['content'])

# # for message in st.session_state['message_history']:
# #     with st.chat_message(message['role']):
# #         st.text(message['content'])

# user_input=st.chat_input('Type here')

# if user_input:
#     st.session_state['message_history'].append({'role':'user','content':user_input})
#     with st.chat_message('user'):
#         st.text(user_input)

    
#     with st.chat_message('assistant'):
#         ai_message=rag_agent.invoke({'messages':[HumanMessage(content=user_input)]},config=config)
#         ai_message_content=ai_message['messages'][-1].content
#         st.text(ai_message_content)
#         # ai_message=st.write_stream(
#         #    message_chunk.content for message_chunk,metadata in  rag_agent.stream(
#         #         {'messages':[HumanMessage(content=user_input)]},
#         #         config=config,
#         #         stream_mode='messages'
#         #     )
#         # )

#         st.session_state['message_history'].append({'role':'assistant','content':ai_message_content})
