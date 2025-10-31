import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. RAG（ナレッジベース）のセットアップ ---

# @st.cache_resource というのは、一度読み込んだ重いデータを
# アプリが再起動しても使いまわせるようにする「おまじない」です。
@st.cache_resource
def setup_rag():
    """
    GitHubからノウハウ(.txt)を読み込み、RAGの検索機能を準備する
    """
    try:
        # --- ノウハウファイルの読み込み ---
        
        # 【！】あなたのGitHubユーザー名/リポジトリ名に修正済み
        repo_owner = "yuuzoozo" # あなたのGitHubユーザー名
        repo_name = "job-posting-mvp" # あなたのリポジトリ名

        # GitHub APIを使ってリポジトリのファイル一覧を取得
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
        response = requests.get(api_url)
        response.raise_for_status() # エラーがあればここで停止
        files = response.json()

        all_documents = []
        
        # .txtファイルだけを対象にループ処理
        for file in files:
            if file['name'].endswith('.txt'):
                # ファイルの「生データ」のURLを取得
                download_url = file['download_url']
                
                # テキストデータをダウンロード
                doc_response = requests.get(download_url)
                doc_response.encoding = 'utf-8' # 文字コードをUTF-8に指定
                
                # 読み込んだテキストデータをリストに追加
                # LangChainが扱いやすいように、 Document オブジェクトとして格納
                from langchain_core.documents import Document
                doc = Document(page_content=doc_response.text, metadata={"source": file['name']})
                all_documents.append(doc)

        if not all_documents:
            st.error("GitHubリポジトリに .txt 形式のノウハウファイルが見つかりません。")
            return None

        # --- 読み込んだノウハウを分割・ベクトル化 ---
        
        # 1. テキスト分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(all_documents)

        # 2. ベクトル化モデル (Gemini)
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                                                    google_api_key=st.secrets["GEMINI_API_KEY"])

        # 3. ベクトルデータベース (Chroma) の構築
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

        # 4. RAGの「検索（Retriever）」機能を返す
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    except Exception as e:
        st.error(f"RAGのセットアップ中にエラーが発生しました: {e}")
        return None

# --- 2. Gemini AI とプロンプトのセットアップ ---

def setup_chain(retriever):
    """
    RAG検索機能とGeminiモデルを連携させ、AIの処理フロー（Chain）を構築する
    """
    try:
        # LLM（AIモデル）の定義
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", 
                                 temperature=0.7, # 創造性（0〜1）
                                 google_api_key=st.secrets["GEMINI_API_KEY"])

        # プロンプト（AIへの指示書）のテンプレート
        # {context} : RAGで検索したノウハウが入る場所
        # {input}   : ユーザーが入力した求人要件が入る場所
        prompt_template = """
        あなたは、日本の採用市場を熟知したプロの採用コンサルタントです。
        以下の「社内ノウハウ」と「求人要件」に基づき、魅力的で具体的な求人票を作成してください。
        
        ## 社内ノウハウ:
        {context}
        
        ## 求人要件:
        {input}
        
        ## アウトプット形式（要件定義書に基づく）:
        必ず以下の形式で、不足している項目も補完しながら作成してください。
        
        ---
        
        ### 募集背景
        （ここに募集背景を具体的に記述）
        
        ### 仕事内容
        （ここに仕事内容を具体的に記述）
        
        ### 応募資格（必須）
        * （ここに必須の応募資格を記述）
        
        ### 応募資格（歓迎）
        * （ここに歓迎の応募資格を記述）
        
        ### この仕事の魅力
        （ここにこの仕事の魅力を具体的に記述）
        
        ---
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        # 1. RAG検索 + プロンプト + LLM をつなぎこむ (Stuff Document Chain)
        #    = 検索結果(context)と要件(input)をプロンプトに埋め込み、LLMに渡す
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # 2. 処理フロー全体を定義する (Retrieval Chain)
        #    = ユーザーの入力(input)をまずRAG検索(retriever)にかけ、
        #      その結果(context)と元の入力(input)を(1)のChainに渡す
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain

    except Exception as e:
        st.error(f"AIチェーンのセットアップ中にエラーが発生しました: {e}")
        return None

# --- 3. Streamlit UI（画面）の構築 ---

st.title("📄 求人票自動生成システム (MVP)")

# APIキーが設定されているかチェック
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Gemini APIキーが設定されていません。Streamlit CloudのSecretsに 'GEMINI_API_KEY' を設定してください。")
else:
    # メインの処理
    try:
        # ステップ1と2を実行
        retriever = setup_rag()
        rag_chain = setup_chain(retriever)

        if retriever and rag_chain:
            st.success("RAGナレッジベースとAIの準備が完了しました。")

            # --- 入力フォーム ---
            with st.form("job_form"):
                st.header("求人要件の入力")
                
                job_title = st.text_input("募集職種", placeholder="例：Webマーケティング（リーダー候補）")
                target_audience = st.text_area("ターゲット層", placeholder="例：30代前半、事業会社でのWebマーケ経験3年以上、将来的にマネジメント志向のある方")
                required_skills = st.text_area("必須スキル（箇条書き）", placeholder="・Web広告（Google/Yahoo）の運用経験\n・SEOの基礎知識")
                welcome_skills = st.text_area("歓迎スキル（箇条書き）", placeholder="・BtoBマーケティングの経験\n・MAツールの運用経験")
                
                # フォームの送信ボタン
                submitted = st.form_submit_button("求人票を自動生成する")

            # --- AIの実行 ---
            if submitted:
                if not job_title:
                    st.warning("「募集職種」は必須入力です。")
                else:
                    with st.spinner("AIが求人票を作成中です..."):
                        try:
                            # ユーザーの入力を1つのテキストにまとめる
                            user_input = f"""
                            募集職種: {job_title}
                            ターゲット層: {target_audience}
                            必須スキル: {required_skills}
                            歓迎スキル: {welcome_skills}
                            """
                            
                            # RAGチェーンを実行（AIが回答を生成）
                            # 'input' というキーで辞書として渡す
                            response = rag_chain.invoke({"input": user_input})
                            
                            # AIの回答（求人票）を表示
                            st.header("生成された求人票")
                            st.markdown(response["answer"]) # 'answer' キーに結果が入っている

                        except Exception as e:
                            st.error(f"求人票の生成中にエラーが発生しました: {e}")

    except Exception as e:
        st.error(f"アプリケーションの起動中に予期せぬエラーが発生しました: {e}")
