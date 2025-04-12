import { Streamlit, RenderData } from "streamlit-component-lib"
import { marked } from "marked"

const COMPONENT_HEIGHT = 450

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, (event: Event) => {
  const data = (event as CustomEvent<RenderData>).detail
  const $component = document.getElementById("component")!

  // We clear the last render, if any.
  $component.innerHTML = ""

  // <sync-id=X>s are interpreted as tags and cause problems,
  // // we replace them with :mark=X:
  let markdown = data.args["markdown"].replace(
    /<sync-id="([^"]*)">/g,
    ":mark=$1:"
  )

  let buffer = (marked(markdown) as string).replace(
    /:mark=([^:]*):/g,
    "<x-mark>$1</x-mark>"
  )
  let doc = new DOMParser().parseFromString(buffer, "text/html")

  let current_mark = add_mark(data.args["first_mark"], $component)
  for (const rootEl of doc.body.childNodes) {
    current_mark = processElement(rootEl as HTMLElement, current_mark)
  }

  Streamlit.setFrameHeight()

  let highlightedMark = data.args["highlighted_mark"]
  if (highlightedMark !== undefined) {
    let parentScrollPos = parent.document.documentElement.scrollTop
    $component
      .querySelector(`[data-timestamp="${highlightedMark}"]`)
      ?.scrollIntoView({ block: "nearest" })
    // Restore scroll pos
    parent.document.documentElement.scrollTop = parentScrollPos
  }
})

Streamlit.setComponentReady()
Streamlit.setFrameHeight()

function processElement(
  el: HTMLElement,
  current_mark: HTMLElement
): HTMLElement {
  if (el.nodeName === "X-MARK") {
    return add_mark(el.innerText, current_mark.parentNode! as HTMLElement)
  }

  if (el.textContent!.trim() === "") {
    return current_mark
  }

  if (el.nodeType === Node.TEXT_NODE) {
    current_mark.innerText += el.textContent
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
  let sub_mark = add_mark(current_mark.dataset.timestamp!, new_el)

  for (const child of el.childNodes) {
    sub_mark = processElement(child as HTMLElement, sub_mark)
  }

  return add_mark(
    sub_mark.dataset.timestamp!,
    current_mark.parentNode! as HTMLElement
  )
}

function add_mark(ts: string, parent: HTMLElement) {
  let mark = document.createElement("span")
  mark.dataset.timestamp = ts
  parent.appendChild(mark)

  return mark
}
