/* Optional UI smoke test against a running local preview backend.
   This signs in, asks the seeded synthetic policy, creates a REVIEW draft, and stops before approval.
   It does not publish a policy. Credentials are supplied only through environment variables. */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const {chromium}=require("playwright");

(async()=>{
  const url=process.env.POLICY_PREVIEW_URL;
  const username=process.env.POLICY_PREVIEW_USERNAME;
  const password=process.env.POLICY_PREVIEW_PASSWORD;
  const output=process.env.POLICY_PREVIEW_SCREENSHOTS;
  if(!url||!username||!password)throw new Error("Set POLICY_PREVIEW_URL, POLICY_PREVIEW_USERNAME, and POLICY_PREVIEW_PASSWORD.");
  if(output)fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:process.env.POLICY_TEST_BROWSER||"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"});
  const page=await browser.newPage({viewport:{width:1440,height:1060}});
  const errors=[];page.on("pageerror",error=>errors.push(error.message));
  const screenshot=async name=>{if(output)await page.screenshot({path:path.join(output,name),fullPage:true});};
  try{
    await page.goto(url);
    await page.locator("#login-form").waitFor({state:"visible"});
    await screenshot("01-login.png");
    await page.locator("#username").fill(username);
    await page.locator("#password").fill(password);
    await page.locator("#login-submit").click();
    await page.locator("#workspace").waitFor({state:"visible"});
    await page.locator("#question").fill("国内出差住宿标准是多少？");
    await page.locator("#ask-submit").click();
    await page.locator(".answer-body").waitFor();
    assert.match(await page.locator(".answer-body").textContent(),/500/);
    assert.equal(await page.locator(".citation-card").count()>0,true);
    await screenshot("02-answer-desktop.png");
    await page.locator('[data-view="documents"]').click();
    await page.locator("#policy-list .policy-row").first().waitFor();
    await screenshot("03-documents-desktop.png");
    await page.locator("#policy-list").getByRole("button",{name:"版本与附件"}).first().click();
    await page.locator(".revision-row").first().waitFor();
    await page.locator(".revision-row").getByRole("button",{name:"查看正文"}).first().click();
    await page.locator(".block-content").first().waitFor();
    assert.match(await page.locator(".block-content").first().textContent(),/500/);
    await screenshot("04-revision.png");
    await page.locator("#close-revisions").click();
    await page.locator('[data-view="changes"]').click();
    await page.locator("#change-instruction").fill("将国内出差住宿标准从每晚 500 元提高到每晚 800 元，其余规则不变。");
    await page.locator("#change-submit").click();
    await page.locator(".diff-after").waitFor();
    assert.match(await page.locator(".diff-before").textContent(),/500/);
    assert.match(await page.locator(".diff-after").textContent(),/800/);
    assert.equal(await page.getByRole("button",{name:"确认修改并发布",exact:true}).isVisible(),true);
    await screenshot("05-review-desktop.png");
    await page.reload();
    await page.locator(".diff-after").waitFor();
    assert.match(await page.locator(".diff-after").textContent(),/800/);
    await page.setViewportSize({width:390,height:844});
    await screenshot("06-review-mobile.png");
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({passed:true,verified:["real cookie login","answer and canonical citation","revision and original/export links","real graph 500→800 draft","review survives reload","390px viewport"],approvalSubmitted:false,screenshots:output||null}));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
