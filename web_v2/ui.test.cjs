/* Browser contract tests. API responses are fixtures; these do not measure backend correctness.
   Run with NODE_PATH pointing to the bundled node_modules, then:
   node --test web_v2/ui.test.cjs
   Set POLICY_TEST_BROWSER to a Chromium executable if Chrome is not in its usual Windows location. */
const {test,before,after}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const http=require("node:http");
const {chromium}=require("playwright");

let server,browser,origin;
const csrf="test-csrf-token";
const actor=(role="manager")=>({id:"user1",username:"Policy Owner",role,scopes:["hr"],active:true});
const policy=(id="p1",title="Travel policy")=>({id,title,scope:"hr",published_revision_id:`r-${id}`,published_revision_number:1});
const session=(role="manager")=>({user:actor(role),csrf_token:csrf,limits:{max_upload_bytes:104857600,max_documents:10000}});

before(async()=>{
  server=http.createServer((request,response)=>{
    const files={"/":"index.html","/static/v2/app.js":"app.js","/static/v2/styles.css":"styles.css"};
    const file=files[request.url.split("?")[0]];
    if(!file){response.writeHead(404);response.end();return;}
    response.setHeader("Content-Type",file.endsWith(".js")?"text/javascript":file.endsWith(".css")?"text/css":"text/html; charset=utf-8");
    response.end(fs.readFileSync(path.join(__dirname,file)));
  });
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  origin=`http://127.0.0.1:${server.address().port}`;
  browser=await chromium.launch({headless:true,executablePath:process.env.POLICY_TEST_BROWSER||"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"});
});
after(async()=>{await browser?.close();if(server)await new Promise(resolve=>server.close(resolve));});

async function setup(role="manager",handler=()=>undefined){
  const context=await browser.newContext({viewport:{width:1365,height:900}});
  await context.addCookies([{name:"test_session",value:"opaque-cookie",url:origin,httpOnly:true,sameSite:"Lax"}]);
  const page=await context.newPage();const calls=[];const errors=[];
  page.on("pageerror",error=>errors.push(error.message));
  await page.route("**/api/v2/**",async route=>{
    const request=route.request();const url=new URL(request.url());
    let body;try{body=request.postDataJSON();}catch{body=request.postData();}
    const call={route:url.pathname.replace("/api/v2",""),query:url.searchParams,method:request.method(),headers:request.headers(),body};calls.push(call);
    let result=await handler(call);
    if(result===undefined){
      if(call.route==="/session"||call.route==="/login")result=session(role);
      else if(call.route==="/policies")result={policies:[policy()]};
      else if(call.route==="/jobs")result={jobs:[]};
      else if(call.route==="/audit")result={events:[]};
      else if(call.route==="/users")result={users:[actor("admin")]};
      else result={};
    }
    const status=result.__status||200;delete result.__status;
    await route.fulfill({status,contentType:"application/json",body:JSON.stringify(result)});
  });
  await page.goto(origin);
  await page.locator("#workspace").waitFor({state:"visible"});
  return {context,page,calls,errors};
}

test("reader sees only authorized flows; untrusted answer and document content stays text",async()=>{
  const hostile='<img src=x onerror="window.injected=true">';
  const fixture=await setup("reader",call=>{
    if(call.route==="/policies")return {policies:[policy("p1",hostile)]};
    if(call.route==="/answers")return {thread_id:"answer-thread",result:{answerable:true,answer:hostile,citations:[{passage_id:"S1",title:hostile,text:hostile,revision_id:"r-p1",revision_number:1,locator:{page:2}}],trace:[{stage:"retrieve",count:1}]}};
  });
  const {page,calls,errors}=fixture;
  try{
    assert.equal(await page.locator('[data-view="changes"]').isVisible(),false);
    assert.equal(await page.locator('[data-view="accounts"]').isVisible(),false);
    await page.locator("#question").fill("What is our transport allowance?");
    await page.locator("#ask-submit").click();
    await page.locator(".answer-body").waitFor();
    assert.equal(await page.locator(".answer-body").textContent(),hostile);
    assert.equal(await page.locator("#answer-result img").count(),0);
    assert.equal(await page.evaluate(()=>window.injected),undefined);
    const answerCall=calls.find(call=>call.route==="/answers");
    assert.equal(answerCall.headers["x-csrf-token"],csrf);
    assert.match(answerCall.headers.cookie,/test_session=opaque-cookie/);
    await page.locator('[data-view="documents"]').click();
    await page.locator("#policy-list").getByText(hostile,{exact:true}).first().waitFor();
    assert.equal(await page.locator("#upload-form").isVisible(),false);
    assert.equal(await page.locator("#policy-list img").count(),0);
    await page.locator(".sidebar .language-toggle").click();
    assert.equal(await page.locator("#page-title").textContent(),"Documents");
    assert.deepEqual(errors,[]);
  }finally{await fixture.context.close();}
});

test("update requires target selection and explicit diff approval; queued work is not published",async()=>{
  let approved=false,published=false;
  const fixture=await setup("manager",call=>{
    if(call.route==="/changes")return {thread_id:"change-1",status:"interrupted",interrupts:[{type:"clarification",question:"Domestic or overseas travel?",candidates:[{id:"candidate-1",title:"Travel policy",heading:"Domestic transport",summary:"Domestic transport allowance",preview:"Current limit: CNY 80 per day."}]}]};
    if(call.route==="/changes/change-1/resume"){
      if(call.body.choice_id)return {thread_id:"change-1",interrupts:[{type:"review",proposal:{id:"proposal-1",policy_title:"Travel policy",scope:"hr",heading:"Domestic transport",base_revision_number:1,before:"Daily allowance: CNY 80.",after:"Daily allowance: CNY 200.",reason:"Update the domestic allowance."}}]};
      approved=true;return {thread_id:"change-1",result:{status:"queued",job:{id:"job-1",status:"queued"}}};
    }
    if(call.route==="/jobs")return {jobs:approved?[{id:"job-1",status:published?"succeeded":"queued",stage:published?"published":"queued",progress:published?100:0}]:[]};
    if(call.route==="/changes/change-1")return {thread_id:"change-1",result:{status:"queued",job:{id:"job-1",status:"queued"}}};
  });
  const {page,calls,errors}=fixture;
  try{
    await page.locator('[data-view="changes"]').click();
    await page.locator("#change-instruction").fill("Change domestic transport allowance to CNY 200.");
    await page.locator("#change-submit").click();
    await page.getByText("Current limit: CNY 80 per day.").waitFor();
    assert.equal(approved,false);
    await page.locator(".candidate-option").click();
    await page.locator(".diff-after").waitFor();
    assert.equal(await page.locator(".diff-before mark").textContent(),"8");
    assert.equal(await page.locator(".diff-after mark").textContent(),"20");
    assert.equal(approved,false);
    await page.getByRole("button",{name:"确认修改并发布",exact:true}).click();
    await page.getByText(/原制度继续生效/).waitFor();
    assert.equal(approved,true);
    assert.equal(await page.getByText("新版本已经发布，知识问答将使用新的制度内容。").count(),0);
    const approval=calls.find(call=>call.body?.approve===true);
    assert.equal(approval.headers["x-csrf-token"],csrf);
    const storage=await page.evaluate(()=>Object.fromEntries(Object.entries(sessionStorage)));
    assert.equal(storage["policy.update.user1"],"change-1");
    assert.equal(Object.values(storage).includes(csrf),false);
    published=true;
    await page.locator('[data-view="documents"]').click();
    await page.locator("#jobs-list").getByText("已完成",{exact:true}).waitFor();
    await page.locator('[data-view="changes"]').click();
    await page.getByText("新版本已经发布，知识问答将使用新的制度内容。").waitFor();
    assert.deepEqual(errors,[]);
  }finally{await fixture.context.close();}
});

test("uploads use multipart CSRF and scoped selections; policy navigation fetches one bounded page",async()=>{
  const fixture=await setup("manager",call=>{
    if(call.route==="/policies"){const offset=Number(call.query.get("offset"));return {policies:offset? [policy("p51","Policy 51")]:Array.from({length:50},(_,index)=>policy(`p${index+1}`,`Policy ${index+1}`))};}
    if(call.route==="/uploads")return {id:"job-upload",status:"queued"};
  });
  const {page,calls,errors}=fixture;
  try{
    await page.locator('[data-view="documents"]').click();
    await page.locator("#policy-list .policy-row").nth(49).waitFor();
    assert.equal(await page.locator("#upload-scope option").allTextContents().then(x=>x.join(",")),"hr");
    assert.equal(await page.locator("#upload-scope-admin").isVisible(),false);
    await page.locator("#policies-next").click();
    await page.getByRole("heading",{name:"Policy 51",exact:true}).waitFor();
    assert.equal(await page.locator("#policy-list .policy-row").count(),1);
    assert.equal(await page.locator("#policies-next").isDisabled(),true);
    assert.equal(calls.filter(call=>call.route==="/policies").every(call=>Number(call.query.get("limit"))===50),true);
    await page.locator("#upload-files").setInputFiles([{name:"one.txt",mimeType:"text/plain",buffer:Buffer.from("A valid policy.")},{name:"two.md",mimeType:"text/markdown",buffer:Buffer.from("# Travel\nA second policy.")}]);
    await page.locator("#upload-submit").click();
    await page.getByText(/已提交 2 个文件/).waitFor();
    const uploads=calls.filter(call=>call.route==="/uploads");
    assert.equal(uploads.length,2);
    for(const call of uploads){assert.equal(call.headers["x-csrf-token"],csrf);assert.match(call.headers["content-type"],/multipart\/form-data/);assert.match(call.body,/name="scope"\r\n\r\nhr/);}
    assert.deepEqual(errors,[]);
  }finally{await fixture.context.close();}
});

test("admin edits roles with CSRF, and mobile layout stays within the viewport",async()=>{
  const fixture=await setup("admin",call=>call.route==="/audit"?{events:[{id:"event1",event:"revision_published",actor_id:"user1",scope:"hr",created_at:1700000000,details:{reason:'<script>window.injected=true</script>'}}]}:undefined);
  const {page,calls,errors}=fixture;
  try{
    await page.locator('[data-view="accounts"]').click();
    await page.locator("#users-list form").waitFor();
    await page.locator("#scopes-user1").fill("hr, finance");
    await page.locator("#users-list form button[type=submit]").click();
    await page.getByText("账号已更新。").waitFor();
    const patch=calls.find(call=>call.method==="PATCH");
    assert.deepEqual(patch.body.scopes,["hr","finance"]);
    assert.equal(patch.headers["x-csrf-token"],csrf);
    await page.locator('[data-view="audit"]').click();
    await page.getByRole("heading",{name:"新版本已发布",exact:true}).waitFor();
    assert.equal(await page.locator("#audit-list script").count(),0);
    assert.match(await page.locator(".audit-row p").textContent(),/Policy Owner/);
    await page.setViewportSize({width:390,height:844});
    await page.locator('[data-view="documents"]').click();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
    assert.deepEqual(errors,[]);
  }finally{await fixture.context.close();}
});

test("failed retry remains actionable, and signing out removes previous account content",async()=>{
  const fixture=await setup("manager",call=>{
    if(call.route==="/policies")return {policies:[policy("private","Private policy text")]};
    if(call.route==="/jobs")return {jobs:[{id:"failed-job",status:"failed",stage:"failed",error:"Parser unavailable"}]};
    if(call.route==="/jobs/failed-job/retry")return {__status:409,detail:"Retry is temporarily unavailable"};
  });
  const {page,calls,errors}=fixture;
  try{
    await page.locator('[data-view="documents"]').click();
    await page.getByRole("heading",{name:"Private policy text",exact:true}).waitFor();
    const retry=page.locator("#jobs-list").getByRole("button",{name:"重试",exact:true});
    await retry.click();
    await page.getByText("Retry is temporarily unavailable",{exact:true}).waitFor();
    assert.equal(await retry.isEnabled(),true);
    await page.locator("#logout").click();
    await page.locator("#login-screen").waitFor({state:"visible"});
    assert.equal(await page.getByText("Private policy text",{exact:true}).count(),0);
    assert.equal(await page.getByText("Parser unavailable",{exact:true}).count(),0);
    assert.equal(calls.find(call=>call.route==="/logout").headers["x-csrf-token"],csrf);
    assert.deepEqual(errors,[]);
  }finally{await fixture.context.close();}
});

test("an approved interrupted publication resumes its existing workflow without drafting again",async()=>{
  let resumed=false;
  const fixture=await setup("manager",call=>{
    if(call.route==="/changes/recover-1"&&call.method==="GET")return {thread_id:"recover-1",status:"recoverable",result:null,interrupts:[],recovery:{type:"publish",message:"Publication can be resumed"}};
    if(call.route==="/changes/recover-1/resume"){
      resumed=true;
      return {thread_id:"recover-1",status:"completed",interrupts:[],result:{status:"queued",job:{id:"existing-job",status:"queued"}}};
    }
    if(call.route==="/jobs")return {jobs:resumed?[{id:"existing-job",status:"queued",stage:"queued",progress:0}]:[]};
  });
  const {page,calls,errors}=fixture;
  try{
    await page.locator('[data-view="changes"]').click();
    await page.evaluate(()=>sessionStorage.setItem("policy.update.user1","recover-1"));
    await page.reload();
    await page.getByRole("heading",{name:"发布流程中断，可继续处理",exact:true}).waitFor();
    assert.equal(await page.locator("#change-content .diff-grid").count(),0);
    assert.equal(await page.locator("#change-content .success-note").count(),0);
    await page.getByRole("button",{name:"继续处理",exact:true}).click();
    await page.locator("#change-content").getByText(/原制度继续生效/).waitFor();
    const resumeCalls=calls.filter(call=>call.route==="/changes/recover-1/resume");
    assert.equal(resumeCalls.length,1);
    assert.deepEqual(resumeCalls[0].body,{approve:true});
    assert.equal(resumeCalls[0].headers["x-csrf-token"],csrf);
    assert.equal(calls.filter(call=>call.route==="/changes"&&call.method==="POST").length,0);
    assert.equal(await page.locator("#change-content").getByText("新版本已经发布，知识问答将使用新的制度内容。").count(),0);
    assert.deepEqual(errors,[]);
  }finally{await fixture.context.close();}
});
