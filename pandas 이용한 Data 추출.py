import json
import sys
from pprint import pprint

sys.stdin.reconfigure(encoding="utf-8")

import pandas as pd
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

client = OpenAI(api_key="sk-")


def find_products(keywords: list[str]) -> list[dict]:
    """제품 설명에 keywords가 포함된 제품을 찾아서 반환합니다."""
    
    print("find_products 함수", keywords)
    df = pd.read_csv("product_list.csv")
    result = df[df["description"].str.contains("|".join(keywords))]
    return result.to_dict(orient="records")


def get_response(chat_log: list, tools: dict = None) -> ChatCompletionMessage:
    
    # completions : llm에게 입력한 대화 기록 기반으로 응답 생성하는 다양한 메서드 제공
    # 모델 답변은 choices[0]에 존재

    response = client.chat.completions.create(model="gpt-4o-mini", messages=chat_log, tools=tools)
    if response.choices[0].message.tool_calls:
        tool_call_results = []
        for tool_call in response.choices[0].message.tool_calls:
            call_id = tool_call.id

            if tool_call.function.name == "find_products":
                keywords = json.loads(tool_call.function.arguments)["keywords"]
                products = find_products(keywords)

                call_result_message = {
                    "tool_call_id": call_id,
                    "role": "tool",
                    "content": json.dumps(
                        {"keywords": keywords, "products": str(products)},
                        ensure_ascii=False,
                    ),
                }
                tool_call_results.append(call_result_message)

        messages = chat_log + [response.choices[0].message] + tool_call_results
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
        )

    return response.choices[0].message

# find_products 함수 기능
# csv 파일에서 > keywords 매개 변수에 들어가 있는 각 문자열 데이터 행을 추출
# find_products_info
# 해당 함수에 필요한 정보들 작성, llm에 넘겨주어 



find_products_info = {
    "name": "find_products",
    "description": "제품 설명에 keywords가 포함된 제품을 찾아서 반환합니다. 찾고자 하는 keywords는 사용자가 요청한 내용과 관련된 제품을 필터링할 수 있는 적절한 키워드여야 합니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "검색하고자 하는 키워드 리스트",  # 1. description을 적절한 내용으로 채우세요.
            },
        },
        "required": ["keywords"],
        "additionalProperties": False,
    },
}

tools = [{"type": "function", "function": find_products_info}]


def main():
    chat_log = [
        {
            "role": "system",
            "content": "너는 쇼핑마트의 AI 챗봇이야. 다양한 제품에 대해 물어보거나 추천을 하는 용도야. 추천을 할 때는 상품 목록에서 사용자가 원하는 상품만 선별해서 추천해야 해.",
        },
    ]

    # None  # 2. chat_log에 사용자가 입력한 대화를 추가하세요
    chat_log.append(
        {
            "role":"user",
            "content":input("You> ")
        }
    )
    
    response = get_response(chat_log, tools=tools)
    print("AI>", response.content)


if __name__ == "__main__":
    main()
