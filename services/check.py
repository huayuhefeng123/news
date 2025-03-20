# from langchain_community.llms import Ollama  # 更新导入路径
# from langchain_core.prompts import ChatPromptTemplate
# import os
# from PIL import Image
# import io
# import base64
# import traceback  # 添加以便打印详细错误信息

# def assess_text_credibility(text, credibility_score):
#     """
#     Assess the credibility of news text using an LLM.
    
#     Parameters:
#     - text (str): The news text to be assessed
#     - credibility_score (float): A score between 0 and 1, where values closer to 1 indicate higher suspicion of falsehood
    
#     Returns:
#     - dict: Contains assessment result, reasoning, and credibility score
#     """
#     try:
#         # 初始化语言模型，使用更新的导入路径
#         llm = Ollama(model="deepseek-r1:1.5b", temperature=0.3)
        
#         # 创建评估提示
#         system_prompt = """
#         [角色] 你是一位专业的新闻事实核查专家，擅长分析新闻报道的真实性和准确性。
        
#         [任务] 对提供的新闻文本进行深入分析，判断它是真实新闻还是虚假新闻，并提供详细的分析理由。
        
#         [参考信息] 
#         - 虚假新闻常见特征：情感化煽动性语言、缺乏具体可验证的信息来源、夸张的断言、时间地点人物不明确、过度使用感叹号和全大写文字、缺乏平衡观点
#         - 真实新闻常见特征：提供具体细节（时间、地点、人物、数据）、引用可验证的信息来源、使用中立客观的语言、提供多方观点、逻辑连贯且合理
#         - 即使是真实新闻，也可能含有错误或偏见；即使是虚假新闻，也可能包含部分真实信息
        
#         [事先怀疑系数] 该新闻的预先怀疑度为 {credibility_score}（0-1之间，越接近1表示越可能是虚假信息）。
        
#         [输出格式] 请严格按照以下格式提供分析:
        
#         ## 新闻真实性分析
        
#         **最终判断**: [明确指出"真实新闻"或"虚假新闻"]
        
#         **详细分析**:
#         1. [详细分析新闻文本的语言特征，包括语气、词汇选择、情感程度等]
#         2. [详细分析新闻的信息来源，包括引用的专家、机构、数据来源等]
#         3. [详细分析新闻的逻辑连贯性和事实一致性]
#         4. [详细分析新闻中的细节呈现，如时间、地点、人物、事件描述等]
#         5. [详细分析新闻的叙事结构和写作手法]
        
#         **关键可疑/可信点**:
#         - [列出文本中最具说服力的真实性指标或最明显的虚假性指标]
        
#         **判断理由总结**:
#         [用3-5句话简洁总结为什么该新闻被判断为真实或虚假]
        
#         [注意事项]
#         1. 分析要全面细致，不要只关注表面特征
#         2. 不要对新闻内容本身发表政治立场或个人观点
#         3. 注意区分客观事实和主观评价
#         4. 分析应具体而非笼统，引用文本中的具体内容作为依据
#         """
        
#         # 直接创建提示并格式化
#         formatted_prompt = system_prompt.format(credibility_score=credibility_score) + "\n\n" + text
        
#         # 直接调用LLM获取响应
#         print("正在调用LLM进行分析...")
#         response = llm.invoke(formatted_prompt)
#         print(f"获得响应: {response}")
        
#         return {
#             "assessment": response,
#         }
            
#     except Exception as e:
#         # 打印详细错误信息以便调试
#         print(f"错误: {str(e)}")
#         traceback.print_exc()
#         return {
#             "assessment": f"分析过程中出现错误: {str(e)}",  # 确保即使出错也返回assessment键
#             "error": str(e),
#         }


# def assess_image_credibility(image_path, credibility_score):
#     """
#     Assess the credibility of an image using a multimodal LLM.
    
#     Parameters:
#     - image_path (str): Path to the image file
#     - credibility_score (float): A score between 0 and 1, where values closer to 1 indicate higher suspicion of falsehood
    
#     Returns:
#     - dict: Contains assessment result, reasoning, and credibility score
#     """
#     try:
#         # 初始化多模态LLM，使用更新的导入路径
#         print(f"正在加载模型以分析图像...")
#         llm = Ollama(model="llava:13b", temperature=0.3)
        
#         # 加载并编码图像
#         print(f"正在加载图像: {image_path}")
#         with open(image_path, "rb") as img_file:
#             img_data = img_file.read()
#             img_base64 = base64.b64encode(img_data).decode("utf-8")
        
#         # 打开图像获取元数据
#         img = Image.open(image_path)
#         width, height = img.size
#         format_type = img.format
#         mode = img.mode
#         print(f"图像信息: {width}x{height}, 格式: {format_type}, 模式: {mode}")
        
#         # 创建详细的图像分析提示
#         image_prompt = f"""
#         [角色] 你是一位专业的图像真实性分析专家，擅长识别图像是否为真实拍摄、经过编辑、AI生成或篡改。

#         [任务] 对提供的图像进行深入分析，明确判断它是真实图像还是虚假图像，并提供详细的视觉分析依据。

#         [图像信息] 图像尺寸: {width}x{height}, 格式: {format_type}, 模式: {mode}

#         [事先怀疑系数] 该图像的预先怀疑度为 {credibility_score}（0-1之间，越接近1表示越可能是虚假或篡改的图像）。

#         [详细分析要点]
#         1. 光影一致性：分析光源方向、阴影细节、高光质量、反射特性是否自然一致
#         2. 边缘分析：检查物体边缘是否自然，有无不自然的锐化、模糊、像素断裂或人工描边
#         3. 透视与比例：评估场景中物体的比例、透视关系、空间深度是否符合现实物理规则
#         4. 纹理分析：检查皮肤、织物、自然元素等纹理是否自然，有无重复模式或异常平滑区域
#         5. 噪点一致性：分析图像不同区域的噪点、颗粒度是否均匀一致
#         6. 颜色分析：检查色彩过渡、颜色分布、饱和度是否自然
#         7. AI生成特征：识别人工智能生成图像的典型特征，如不自然的细节、奇怪的物体融合、结构错误等
#         8. 元素完整性：检查人物、物体的完整性，如手指数量、牙齿、眼睛、发际线等细节

#         [输出格式] 请严格按照以下格式提供分析:

#         ## 图像真实性分析

#         **最终判断**: [明确指出"真实图像"或"虚假图像"]

#         **详细视觉分析**:
#         1. [光影分析：详细分析图像中的光线、阴影、高光等是否自然，是否存在不一致]
#         2. [边缘与纹理：详细分析图像中的边缘、纹理特征，特别是人物或主体物件]
#         3. [透视与比例：详细分析图像中的透视关系和物体比例是否符合物理规则]
#         4. [不自然元素：详细指出图像中任何不符合现实逻辑的元素或特征]
#         5. [技术特征：分析图像的技术特征，如噪点分布、压缩痕迹、像素特征等]

#         **关键可疑/可信区域**:
#         - [具体描述图像中最能证明其真实性或虚假性的关键区域和细节]

#         **判断理由总结**:
#         [用3-5句话简洁总结为什么该图像被判断为真实或虚假]

#         [注意事项]
#         1. 分析必须基于具体的视觉特征，避免笼统的描述
#         2. 清晰区分确定性观察和推测性判断
#         3. 如有不确定之处，应明确指出并解释原因
#         """
        
#         # 调用模型
#         print("正在调用模型分析图像...")
#         response = llm.invoke(image_prompt)
#         print(f"获得响应: {response}...")  # 打印响应的前100个字符，用于调试
        
#         return {
#             "assessment": response,
#             "input_credibility_score": credibility_score
#         }
            
#     except Exception as e:
#         # 打印详细错误信息以便调试
#         print(f"图像分析错误: {str(e)}")
#         traceback.print_exc()
#         return {
#             "assessment": f"图像分析过程中出现错误: {str(e)}",  # 确保即使出错也返回assessment键
#             "error": str(e),
#             "input_credibility_score": credibility_score
#         }


# # 示例用法
# if __name__ == "__main__":
#     # 示例文本评估
#     sample_news = """
#     突发！某城市发生严重洪水，已造成数千人伤亡！政府部门掩盖真相，拒绝公布真实数据！
#     一位不愿透露姓名的知情人士透露，实际情况比报道的更加严重。网友纷纷在社交媒体上传播相关图片，
#     引发公众恐慌。专家表示这是近百年来最严重的灾难，但官方媒体却几乎没有报道！
#     """
    
#     print("\n开始评估文本真实性...")
#     text_result = assess_text_credibility(sample_news, 0.8)
#     print("\n===== 文本真实性评估结果 =====")
#     print(text_result["assessment"])
    
#     # 示例图像评估 - 取消注释以测试实际图像
#     # print("\n开始评估图像真实性...")
#     # image_result = assess_image_credibility("path_to_image.jpg", 0.7)
#     # print("\n===== 图像真实性评估结果 =====")
#     # print(image_result["assessment"])

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import os
from PIL import Image
import io
import base64
import traceback
import time
import gc  # 添加垃圾回收模块

def release_resources():
    """释放内存资源"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA缓存已清空")
    except ImportError:
        print("PyTorch未安装，跳过CUDA清理")

def assess_text_credibility(text, credibility_score, retry_count=3, model="deepseek-r1:1.5b", fallback_model="phi2"):
    """
    Assess the credibility of news text using an LLM.
    添加了重试逻辑和备用模型选项
    """
    for attempt in range(retry_count):
        try:
            current_model = model if attempt == 0 else fallback_model
            print(f"尝试使用模型: {current_model} (尝试 {attempt+1}/{retry_count})")
            
            # 初始化语言模型
            llm = Ollama(model=current_model, temperature=0.3)
            
            # 创建评估提示
            system_prompt = """
            [角色] 你是一位专业的新闻事实核查专家，擅长分析新闻报道的真实性和准确性。
            
            [任务] 对提供的新闻文本进行深入分析，判断它是真实新闻还是虚假新闻，并提供详细的分析理由。
            
            [参考信息] 
            - 虚假新闻常见特征：情感化煽动性语言、缺乏具体可验证的信息来源、夸张的断言、时间地点人物不明确、过度使用感叹号和全大写文字、缺乏平衡观点
            - 真实新闻常见特征：提供具体细节（时间、地点、人物、数据）、引用可验证的信息来源、使用中立客观的语言、提供多方观点、逻辑连贯且合理
            - 即使是真实新闻，也可能含有错误或偏见；即使是虚假新闻，也可能包含部分真实信息
            
            [事先怀疑系数] 该新闻的预先怀疑度为 {credibility_score}（0-1之间，越接近1表示越可能是虚假信息）。
            
            [输出格式] 请严格按照以下格式提供分析:
            
            ## 新闻真实性分析
            
            **最终判断**: [明确指出"真实新闻"或"虚假新闻"]
            
            **详细分析**:
            1. [详细分析新闻文本的语言特征，包括语气、词汇选择、情感程度等]
            2. [详细分析新闻的信息来源，包括引用的专家、机构、数据来源等]
            3. [详细分析新闻的逻辑连贯性和事实一致性]
            4. [详细分析新闻中的细节呈现，如时间、地点、人物、事件描述等]
            5. [详细分析新闻的叙事结构和写作手法]
            
            **关键可疑/可信点**:
            - [列出文本中最具说服力的真实性指标或最明显的虚假性指标]
            
            **判断理由总结**:
            [用3-5句话简洁总结为什么该新闻被判断为真实或虚假]
            """
            
            # 直接创建提示并格式化
            formatted_prompt = system_prompt.format(credibility_score=credibility_score) + "\n\n" + text
            
            # 调用LLM获取响应
            print("正在调用LLM进行分析...")
            response = llm.invoke(formatted_prompt)
            print(f"获得响应: {response[:100]}...")  # 仅打印响应开头部分
            
            # 释放资源
            del llm
            release_resources()
            
            return {
                "assessment": response,
            }
                
        except Exception as e:
            print(f"尝试 {attempt+1} 失败: {str(e)}")
            if "unable to allocate CUDA" in str(e) or "500" in str(e):
                print("GPU内存不足，尝试释放资源并等待...")
                release_resources()
                time.sleep(5)  # 等待系统释放资源
            
            # 如果是最后一次尝试，则返回错误信息
            if attempt == retry_count - 1:
                return {
                    "assessment": f"分析过程中出现错误: {str(e)}",
                    "error": str(e),
                }
                
            # 否则继续下一次尝试
            continue


def assess_image_credibility(image_path, credibility_score, retry_count=3, 
                            model="llava:13b", fallback_models=["bakllava:7b", "llava:7b"]):
    """
    Assess the credibility of an image using a multimodal LLM.
    添加了重试逻辑、备用模型选项和内存管理
    """
    # 首先获取图像信息，这不需要模型加载
    try:
        print(f"正在加载图像: {image_path}")
        # 加载并编码图像
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode("utf-8")
        
        # 打开图像获取元数据
        img = Image.open(image_path)
        width, height = img.size
        format_type = img.format
        mode = img.mode
        print(f"图像信息: {width}x{height}, 格式: {format_type}, 模式: {mode}")
    except Exception as e:
        print(f"图像加载错误: {str(e)}")
        return {
            "assessment": f"图像加载过程中出现错误: {str(e)}",
            "error": str(e),
            "input_credibility_score": credibility_score
        }
    
    # 创建详细的图像分析提示
    image_prompt = f"""
    [角色] 你是一位专业的图像真实性分析专家，擅长识别图像是否为真实拍摄、经过编辑、AI生成或篡改。

    [任务] 对提供的图像进行深入分析，明确判断它是真实图像还是虚假图像，并提供详细的视觉分析依据。

    [图像信息] 图像尺寸: {width}x{height}, 格式: {format_type}, 模式: {mode}

    [事先怀疑系数] 该图像的预先怀疑度为 {credibility_score}（0-1之间，越接近1表示越可能是虚假或篡改的图像）。

    [详细分析要点]
    1. 光影一致性：分析光源方向、阴影细节、高光质量、反射特性是否自然一致
    2. 边缘分析：检查物体边缘是否自然，有无不自然的锐化、模糊、像素断裂或人工描边
    3. 透视与比例：评估场景中物体的比例、透视关系、空间深度是否符合现实物理规则
    4. 纹理分析：检查皮肤、织物、自然元素等纹理是否自然，有无重复模式或异常平滑区域
    5. 噪点一致性：分析图像不同区域的噪点、颗粒度是否均匀一致
    6. 颜色分析：检查色彩过渡、颜色分布、饱和度是否自然
    7. AI生成特征：识别人工智能生成图像的典型特征，如不自然的细节、奇怪的物体融合、结构错误等
    8. 元素完整性：检查人物、物体的完整性，如手指数量、牙齿、眼睛、发际线等细节

    [输出格式] 请严格按照以下格式提供分析:

    ## 图像真实性分析

    **最终判断**: [明确指出"真实图像"或"虚假图像"]

    **详细视觉分析**:
    1. [光影分析：详细分析图像中的光线、阴影、高光等是否自然，是否存在不一致]
    2. [边缘与纹理：详细分析图像中的边缘、纹理特征，特别是人物或主体物件]
    3. [透视与比例：详细分析图像中的透视关系和物体比例是否符合物理规则]
    4. [不自然元素：详细指出图像中任何不符合现实逻辑的元素或特征]
    5. [技术特征：分析图像的技术特征，如噪点分布、压缩痕迹、像素特征等]

    **关键可疑/可信区域**:
    - [具体描述图像中最能证明其真实性或虚假性的关键区域和细节]

    **判断理由总结**:
    [用3-5句话简洁总结为什么该图像被判断为真实或虚假]
    """

    # 尝试使用不同模型
    all_models = [model] + fallback_models
    
    for attempt, current_model in enumerate(all_models[:retry_count]):
        try:
            print(f"尝试使用模型: {current_model} (尝试 {attempt+1}/{min(retry_count, len(all_models))})")
            
            # 在每次尝试前清理内存
            release_resources()
            
            # 初始化多模态LLM
            llm = Ollama(model=current_model, temperature=0.3)
            
            # 调用模型
            print("正在调用模型分析图像...")
            response = llm.invoke(image_prompt)
            print(f"获得响应: {response[:100]}...")  # 仅打印响应开头部分
            
            # 释放资源
            del llm
            release_resources()
            
            return {
                "assessment": response,
                "input_credibility_score": credibility_score
            }
                
        except Exception as e:
            print(f"尝试 {attempt+1} 使用 {current_model} 失败: {str(e)}")
            if "unable to allocate CUDA" in str(e) or "500" in str(e):
                print(f"GPU内存不足，释放资源并等待...")
                release_resources()
                time.sleep(10)  # 给系统更多时间释放资源
            
            # 如果是最后一次尝试，则返回错误信息
            if attempt == min(retry_count, len(all_models)) - 1:
                return {
                    "assessment": f"图像分析过程中出现错误: {str(e)}",
                    "error": str(e),
                    "input_credibility_score": credibility_score
                }
                
            # 否则继续下一次尝试
            continue


# 示例用法
if __name__ == "__main__":
    # 示例文本评估
    sample_news = """
    突发！某城市发生严重洪水，已造成数千人伤亡！政府部门掩盖真相，拒绝公布真实数据！
    一位不愿透露姓名的知情人士透露，实际情况比报道的更加严重。网友纷纷在社交媒体上传播相关图片，
    引发公众恐慌。专家表示这是近百年来最严重的灾难，但官方媒体却几乎没有报道！
    """
    
    print("\n开始评估文本真实性...")
    text_result = assess_text_credibility(sample_news, 0.8)
    print("\n===== 文本真实性评估结果 =====")
    print(text_result["assessment"])
    
    # 在函数调用之间强制清理资源
    release_resources()
    time.sleep(5)  # 等待系统处理
    
    # 示例图像评估 - 取消注释以测试实际图像
    # print("\n开始评估图像真实性...")
    # image_result = assess_image_credibility("path_to_image.jpg", 0.7)
    # print("\n===== 图像真实性评估结果 =====")
    # print(image_result["assessment"])