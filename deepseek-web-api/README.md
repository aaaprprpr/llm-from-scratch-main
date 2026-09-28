# DeepSeek 与千问网页聊天请求笔记

记录时间：2026-09-28。两套请求都来自已登录网页的网络请求，属于**网页内部接口**，服务方可能调整字段、签名或风控。聊天脚本 [web_chat_once.mjs](web_chat_once.mjs) 与 [qwen_chat.mjs](qwen_chat.mjs) 在 Node.js 18+ 中运行，不需要安装 npm 包。

## 直接使用

项目的 `.env` 已配置 DeepSeek 令牌和千问凭据文件路径；千问的已登录 Cookie 与签名材料保存在被 `.gitignore` 忽略的 `.qwen-web-auth.json`。当前环境可直接运行：

```powershell
node .\web_chat_once.mjs '你好'   # DeepSeek：首次新建，之后自动续聊
node .\qwen_chat.mjs '你好'       # 千问：首次新建，之后自动续聊
```

两种脚本使用相同的会话操作：

| 操作 | DeepSeek 示例 | 千问示例 |
| --- | --- | --- |
| 发送并自动复用上次会话 | `node .\web_chat_once.mjs '下一句'` | `node .\qwen_chat.mjs '下一句'` |
| 明确新建会话 | `node .\web_chat_once.mjs --new '你好'` | `node .\qwen_chat.mjs --new '你好'` |
| 明确继续已有会话 | `node .\web_chat_once.mjs --continue '下一句'` | `node .\qwen_chat.mjs --continue '下一句'` |
| 删除当前保存的会话 | `node .\web_chat_once.mjs --delete-current` | `node .\qwen_chat.mjs --delete-current` |
| 只问一次并立即删除本次新会话 | `node .\web_chat_once.mjs --delete-after '你好'` | `node .\qwen_chat.mjs --delete-after '你好'` |

`--delete-after` 默认创建临时新会话，不覆盖已有续聊状态。`--continue --delete-after` 会使用并删除当前保存的会话。`--new` 会将本地“当前会话”切换为新会话；旧会话仍留在服务端，因此若不需要旧会话，先执行 `--delete-current`。

### 会话 ID 到底怎么用

| 网站 | 会话 ID | 后续消息需要的另一个 ID | 本地状态文件 |
| --- | --- | --- | --- |
| DeepSeek | `chat_session_id`：创建接口返回的 `data.biz_data.id` | `parent_message_id`：上一轮助手回复 SSE `event: ready` 中的 `response_message_id` | `.deepseek-session.json` |
| 千问 | `session_id`：客户端生成的 32 位十六进制 ID，首轮请求即携带 | `parent_req_id`：上一轮**请求**的 `req_id`，首轮为字符串 `"0"` | `.qwen-session.json` |

删除会话只传对应的**会话 ID**；上一轮消息 ID / 请求 ID 是续聊用的。示例脚本自动保存和更新这些 ID，无需手动复制。两个状态文件和凭据文件均被 `.gitignore` 忽略。请勿把状态文件当作多会话列表：每个网站只保存一个“当前会话”。

## DeepSeek 网页接口

观察对象是已登录的 `https://chat.deepseek.com/`，当时前端版本为 `2.5.0`。

### 最短调用链

1. 用网页登录令牌调用 `POST /api/v0/chat_session/create`，JSON 请求体 `{}`。会话 ID 在响应的 `data.biz_data.id`。
2. 调用 `POST /api/v0/chat/create_pow_challenge`，请求体 `{"target_path":"/api/v0/chat/completion"}`。挑战在 `data.biz_data.challenge`。
3. 解出 `DeepSeekHashV1` 挑战，构造 `X-DS-PoW-Response`。
4. 调用 `POST /api/v0/chat/completion`，返回 Server-Sent Events（SSE）流。

三个地址的主机都是 `https://chat.deepseek.com`。前两个请求实测只需 `Authorization: Bearer <网页登录令牌>` 和 `Content-Type: application/json`。第三个还需 `X-DS-PoW-Response`。我在已登录浏览器中以 `credentials: 'omit'` 重放，仅带这三个请求头，得到了 `text/event-stream` 和预期回复；因此该次测试不依赖 Cookie、`x-client-*`、`x-device-*` 或 `x-hif-*` 请求头。独立 Node 进程的网络请求仍可能遇到额外风控。

### 请求格式

#### 1. 创建会话

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

#### 2. 获取 PoW 挑战

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

#### 3. 发送一条消息

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

### 运行环境与配置

- Node.js 18 或更新版本；使用内置 `fetch`，无需 `npm install`。
- 机器能通过 HTTPS 访问 `chat.deepseek.com`，并能正常登录网页取得 `WEB_TOKEN`。脚本运行时不需要启动浏览器。
- 登录令牌是**网页账号令牌**，与 `platform.deepseek.com` 的正式 API Key 不同。项目的 `.env` 是明文文件，已列入 `.gitignore`；不要截图、分享或提交它。过期后需从网页重新获取。

当前项目的 `.env` 已从你登录的浏览器写入，脚本会自动读取。直接运行：

```powershell
node .\web_chat_once.mjs '请只回复 OK'
```

以后令牌失效时，在已登录的网页按 F12，进入 Console，执行以下命令把新令牌复制到剪贴板（**不要将结果发给别人**）：

```js
copy(JSON.parse(localStorage.getItem('userToken')).value)
```

然后将 `.env` 中的 `DEEPSEEK_WEB_TOKEN` 值替换为新令牌。也可以临时设置同名环境变量，它会覆盖 `.env`：

```powershell
$secret = Read-Host '粘贴网页登录令牌' -AsSecureString
$env:DEEPSEEK_WEB_TOKEN = [System.Net.NetworkCredential]::new('', $secret).Password
node .\web_chat_once.mjs '请只回复 OK'
Remove-Item Env:DEEPSEEK_WEB_TOKEN
```

预期输出是 `OK`。首次运行创建会话，后续直接运行会继续本地保存的会话。脚本只使用 HTTP 请求，运行时不依赖浏览器。首次调用包含 PoW 计算，耗时取决于挑战难度和 CPU。

### 删除会话

```http
POST /api/v0/chat_session/delete HTTP/1.1
Host: chat.deepseek.com
Authorization: Bearer <WEB_TOKEN>
Content-Type: application/json

{"chat_session_id":"<SESSION_ID>"}
```

成功响应包含 `code: 0` 与 `data.biz_code: 0`。已通过“创建测试会话 → 删除测试会话”及脚本 `--delete-after` 实测。`--delete-current` 从 `.deepseek-session.json` 读取 ID 后调用此接口。

### 范围与注意点

- 以上由当前网页请求、只带最少请求头的重放，以及独立 Node 示例的生产站点请求验证。Node 示例在本机实际收到了 `NODE_OK`。
- 网页聊天接口的登录令牌、模型名、PoW、风控及速率限制都可能调整。收到 JSON `code` 非零、401/403、验证码或其他挑战时，应先对照网页最新网络请求；不要把旧 PoW 反复重放。
- 若要长期维护或发布工具，优先考虑 DeepSeek [官方 Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)；它使用独立 API Key 和 `https://api.deepseek.com/chat/completions`，请求格式与这里的网页内部接口不同。

## 千问网页接口

观察对象是已登录的 `https://www.qianwen.com/`；本次抓到的网页资源版本为 `@ali/qianwen-web/4.9.0`。它与阿里云百炼的[正式千问 API](https://help.aliyun.com/zh/model-studio/qwen-api-reference)是两套接口。

### 环境与凭据

- 聊天脚本：Node.js 18+，直接发 HTTPS 请求，不需要浏览器常驻。
- 首次导出或更新网页凭据：已登录且启用远程调试的 360 浏览器、Python 3、`websocket-client`（缺少时运行 `python -m pip install websocket-client`）。
- 当前调试浏览器端口是 `53267`；若重开浏览器后端口变化，应把命令中的端口换成新的。网页先加载完会话列表，再运行：

```powershell
python .\qwen_capture_auth.py 53267
node .\qwen_chat.mjs '请只回复 OK'
```

导出脚本通过本机 Chrome DevTools Protocol 读取登录 Cookie、网页 `localStorage` 中的 `qwen_chat` 签名材料以及公共查询参数，写入 `.qwen-web-auth.json`，终端只打印项目数量。`.env` 中的 `QWEN_WEB_AUTH_FILE` 指向该文件；默认路径也就是项目根目录的 `.qwen-web-auth.json`。Cookie、`ut` 和签名材料均属于网页登录凭据，不要分享或提交文件。签名材料余量低或将过期时，聊天脚本会自动补充；两个批量槽共用凭据文件锁，并按至少 2 秒的间隔启动请求。补充也失败时，检查网页登录状态并重新导出。

### 发消息

```http
POST /api/v2/chat?<公共查询参数>&nonce=<随机值>&timestamp=<毫秒时间戳> HTTP/1.1
Host: chat2.qianwen.com
Cookie: <已登录网页 Cookie>
Content-Type: application/json
clt-acs-sign: <动态 HMAC 签名>
clt-acs-bfg: <请求体 HMAC>
clt-acs-reqt: <毫秒时间戳>
clt-acs-request-params: <参与签名的查询参数名，按原顺序连接>
eo-clt-actkn: <qwen_chat 场景令牌>
eo-clt-sacsft: <本次签名材料>
...

{"req_id":"<新生成的 32 位十六进制 ID>","parent_req_id":"0","messages":[{"mime_type":"text/plain","content":"你好","meta_data":{"ori_query":"你好"},"status":"complete"}],"scene":"chat","scene_param":"first_turn","session_id":"<新生成的 32 位十六进制会话 ID>","biz_id":"ai_qwen","model":"Qwen","protocol_version":"v2","chat_client":"h5","chat_mode":"quick"}
```

示意请求体只展示关键字段；[qwen_chat.mjs](qwen_chat.mjs) 含当前网页需要的完整字段。公共查询参数包括 `biz_id`、`chat_client`、`device`、`fr`、`pr`、`ut`、`la`、`tz`、`wv`、`tq`、`ve`，聊天请求另加 `fe_version=1.0.0`。签名算法的关键步骤：取 `qwen_chat` 的一个 `eo-clt-bacsft` 值，拼接 `secret = sacsft + ":" + timestamp`；请求体的 HMAC-SHA256 Base64 是 `clt-acs-bfg`；再对设备 ID、签名版本 `1.0.0`、按顺序拼接的查询参数值、`tongyi_sso_ticket_hash` 和请求体 HMAC 拼接后的字符串做一次 HMAC-SHA256 Base64，得到 `clt-acs-sign`。签名同时依赖查询参数的顺序和请求体的精确 JSON 字节。

首轮 `parent_req_id: "0"`、`scene_param: "first_turn"`。续聊时保持同一 `session_id`，把 `parent_req_id` 设为**上一轮请求的 `req_id`**，`scene_param` 变为 `"continue_chat"`。响应是 SSE；本次普通文本回复出现在 `data.messages[]` 的 `mime_type: "multi_load/iframe"` 项，后面可能有对前面内容的更新，脚本取最后的完整内容。只检查 HTTP 200 不够：缺少正确签名时，本次测试曾收到 200，但仅返回固定拒答。

### 删除会话

```http
POST /api/v1/session/delete?<公共查询参数> HTTP/1.1
Host: chat2-api.qianwen.com
Cookie: <已登录网页 Cookie>
Content-Type: application/json
X-XSRF-TOKEN: <XSRF-TOKEN Cookie 的值>

{"session_id":"<SESSION_ID>","biz_id":"ai_qwen"}
```

公共查询参数中的 `ut` 必须带上；只发路径和 JSON 请求体时实测返回 HTTP 400、`msg: "ut"`。带上网页的公共查询参数后，返回 HTTP 200、`success: true`、`code: 0`。已用本次生成的测试会话验证删除，脚本也实测了 `--delete-current` 和 `--delete-after`。

### 版本限制

这套网页接口和签名材料会跟随千问网页升级。若出现固定拒答、SSE 为空、401/403 或签名材料过期，先重新运行导出脚本；仍失败时，对照浏览器最新的 `/api/v2/chat` 请求更新字段。需要稳定长期集成时，使用[阿里云百炼正式 API](https://help.aliyun.com/zh/model-studio/first-api-call-to-qwen)。
