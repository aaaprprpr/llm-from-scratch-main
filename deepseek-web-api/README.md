# DeepSeek 网页聊天请求笔记

记录时间：2026-09-28。观察对象是已登录的 `https://chat.deepseek.com/` 网页，前端显示版本 `2.5.0`。这是**网页内部接口**，并非 DeepSeek 承诺兼容的公开 API；字段、PoW 和风控要求可能更新。项目中的 [web_chat_once.mjs](web_chat_once.mjs) 是零第三方依赖的一次请求示例。

## 最短调用链

1. 用网页登录令牌调用 `POST /api/v0/chat_session/create`，JSON 请求体 `{}`。会话 ID 在响应的 `data.biz_data.id`。
2. 调用 `POST /api/v0/chat/create_pow_challenge`，请求体 `{"target_path":"/api/v0/chat/completion"}`。挑战在 `data.biz_data.challenge`。
3. 解出 `DeepSeekHashV1` 挑战，构造 `X-DS-PoW-Response`。
4. 调用 `POST /api/v0/chat/completion`，返回 Server-Sent Events（SSE）流。

三个地址的主机都是 `https://chat.deepseek.com`。前两个请求实测只需 `Authorization: Bearer <网页登录令牌>` 和 `Content-Type: application/json`。第三个还需 `X-DS-PoW-Response`。我在已登录浏览器中以 `credentials: 'omit'` 重放，仅带这三个请求头，得到了 `text/event-stream` 和预期回复；因此该次测试不依赖 Cookie、`x-client-*`、`x-device-*` 或 `x-hif-*` 请求头。独立 Node 进程的网络请求仍可能遇到额外风控。

## 请求格式

### 1. 创建会话

```http
POST /api/v0/chat_session/create HTTP/1.1
Host: chat.deepseek.com
Authorization: Bearer <WEB_TOKEN>
Content-Type: application/json

{}
```

成功响应的结构：

```json
{"code":0,"data":{"biz_code":0,"biz_data":{"id":"<SESSION_ID>","current_message_id":null}}}
```

### 2. 获取 PoW 挑战

```http
POST /api/v0/chat/create_pow_challenge HTTP/1.1
Host: chat.deepseek.com
Authorization: Bearer <WEB_TOKEN>
Content-Type: application/json

{"target_path":"/api/v0/chat/completion"}
```

`data.biz_data.challenge` 包含 `algorithm`、`challenge`、`salt`、`signature`、`difficulty`、`expire_at`、`expire_after`、`target_path`。当前 `algorithm` 是 `DeepSeekHashV1`。前端会从 `0` 到 `difficulty - 1` 寻找整数 `answer`，使其内部哈希对字符串 `salt + "_" + expire_at + "_" + answer` 的十六进制结果等于 `challenge`。**这里不能直接用标准 `SHA3-256`**：当前前端的 Keccak 实现执行第 1～23 轮，省略第 0 轮。[pow23.mjs](pow23.mjs) 实现了这一版本，并与当前网页 worker 的结果比对过。

PoW 请求头的值是下面这个 JSON 的 **Base64 编码**，不是原始 JSON：

```json
{"algorithm":"DeepSeekHashV1","challenge":"<挑战值>","salt":"<盐值>","answer":123,"signature":"<签名>","target_path":"/api/v0/chat/completion"}
```

挑战有有效期，发送消息前应现取现算。将原始 JSON 直接放进请求头，实测返回 `{"code":40300,"msg":"MISSING_HEADER"}`。

### 3. 发送一条消息

```http
POST /api/v0/chat/completion HTTP/1.1
Host: chat.deepseek.com
Authorization: Bearer <WEB_TOKEN>
Content-Type: application/json
X-DS-PoW-Response: <上一步 JSON 的 Base64>

{"chat_session_id":"<SESSION_ID>","parent_message_id":null,"model_type":"default","prompt":"你好","ref_file_ids":[],"thinking_enabled":false,"search_enabled":false,"action":null,"preempt":false}
```

新会话首条消息用 `parent_message_id: null`、`model_type: "default"`。同一会话继续聊天时，实测 `parent_message_id` 取上一轮助手消息 ID（例如 SSE `event: ready` 的 `response_message_id`），网页后续轮次会传 `model_type: null`。每次发送前重新取 PoW 挑战更稳妥。

响应 `Content-Type` 为 `text/event-stream`。首帧例如：

```text
event: ready
data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}

data: {"v":{"response":{"fragments":[{"type":"RESPONSE","content":"你好"}]}}}

data: {"p":"response/fragments/-1/content","o":"APPEND","v":"！"}

event: close
data: {"click_behavior":"none","auto_resume":false}
```

回复可能分多帧，`APPEND` 帧按顺序追加；后续帧有时只带 `v`，沿用上一个 `p` 路径。示例脚本处理了这两种情况。

## 运行环境与配置

- Node.js 18 或更新版本；使用内置 `fetch`，无需 `npm install`。
- 机器能通过 HTTPS 访问 `chat.deepseek.com`，并能正常登录网页取得 `WEB_TOKEN`。脚本运行时不需要启动浏览器。
- 登录令牌是**网页账号令牌**，与 `platform.deepseek.com` 的正式 API Key 不同。不要放进源代码、`.env`、截图、日志或 Git；过期后需从网页重新获取。

在已登录的网页按 F12，进入 Console，执行以下命令把令牌复制到剪贴板（**不要将结果发给别人**）：

```js
copy(JSON.parse(localStorage.getItem('userToken')).value)
```

PowerShell 中以不写入命令历史的方式输入令牌，再运行一次请求：

```powershell
$secret = Read-Host '粘贴网页登录令牌' -AsSecureString
$env:DEEPSEEK_WEB_TOKEN = [System.Net.NetworkCredential]::new('', $secret).Password
node .\web_chat_once.mjs '请只回复 OK'
Remove-Item Env:DEEPSEEK_WEB_TOKEN
```

预期输出是 `OK`。脚本每次创建一个新会话，直接使用 HTTP 请求，没有调用浏览器 UI。首次调用包含 PoW 计算，耗时取决于挑战难度和 CPU。

## 范围与注意点

- 以上由当前网页请求、只带最少请求头的重放，以及独立 Node 示例的生产站点请求验证。Node 示例在本机实际收到了 `NODE_OK`。
- 网页聊天接口的登录令牌、模型名、PoW、风控及速率限制都可能调整。收到 JSON `code` 非零、401/403、验证码或其他挑战时，应先对照网页最新网络请求；不要把旧 PoW 反复重放。
- 若要长期维护或发布工具，优先考虑 DeepSeek [官方 Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)；它使用独立 API Key 和 `https://api.deepseek.com/chat/completions`，请求格式与这里的网页内部接口不同。
