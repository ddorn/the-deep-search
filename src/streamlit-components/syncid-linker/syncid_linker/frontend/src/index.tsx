import { Streamlit, RenderData } from "streamlit-component-lib"
import { marked } from "marked"

const $component = document.getElementById("component")

function new_mark(ts: string): HTMLElement {
  let mark = document.createElement("span")
  mark.dataset.timestamp = ts
  return mark
}
function add_mark(ts: string, parent: HTMLElement) {
  let mark = new_mark(ts)
  parent.appendChild(mark)
  return mark
}

function onRender(event: Event): void {
  const data = (event as CustomEvent<RenderData>).detail

  let raw = data.args["markdown"]

  let regex = /<sync-id="([^"]*)">/g
  let replacement = '<ts-mark value="$1"></ts-mark>'
  let buffer = raw.replace(regex, replacement)

  console.log(buffer)

  $container.innerHTML = ""
  let doc = new DOMParser().parseFromString(
    marked(buffer) as string,
    "text/html"
  )
  let initial_ts_mark = document.createElement("ts-mark")
  initial_ts_mark.setAttribute("value", "0")

  let current_mark = add_mark("0", $container)

  for (const rootEl of doc.body.childNodes) {
    current_mark = processElement(rootEl as HTMLElement, current_mark)
  }

  Streamlit.setFrameHeight()
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)

Streamlit.setComponentReady()

Streamlit.setFrameHeight()

function processElement(
  el: HTMLElement,
  current_mark: HTMLElement
): HTMLElement {
  if (el.nodeName === "TS-MARK") {
    let ts = el.getAttribute("value")!
    return add_mark(ts, current_mark.parentNode! as HTMLElement)
  }

  if (el.textContent!.trim() === "") {
    return current_mark
  }

  if (el.nodeType === Node.TEXT_NODE) {
    current_mark.innerText += el.textContent
    // A text node has no children.
    return current_mark
  }

  //console.log(current_mark.dataset.timestamp, el.nodeName)
  // we have an element, so, we create this element
  // on the parent of current_mark, but without its children,
  // then, we create a new mark with the same ts as previous mark,
  //  and as a child of the parent
  //  and, for each child we simply call processElement with the new current_mark
  //
  // Clone el without children
  let new_el = el.cloneNode(false) as HTMLElement
  current_mark.parentNode!.appendChild(new_el)
  let sub_mark = add_mark(get_ts(current_mark), new_el)

  for (const child of el.childNodes) {
    sub_mark = processElement(child as HTMLElement, sub_mark)
  }

  let sub_ts = get_ts(sub_mark)
  return add_mark(sub_ts, current_mark.parentNode! as HTMLElement)
}

function get_ts(el: HTMLElement): string {
  return el.dataset.timestamp!
}
