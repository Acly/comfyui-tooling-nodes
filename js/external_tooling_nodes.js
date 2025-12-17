import { app } from "/scripts/app.js"
import { api } from "/scripts/api.js"

(function() {

// Workflow publishing
// - done whenever the graph changes, as long as there is a KritaOutput node

let publisherRegistered = false

async function publishWorkflow(e) {
    const prompt = await app.graphToPrompt()
    await api.fetchApi("/api/etn/workflow/publish", {
        method: "POST",
        body: JSON.stringify({
            name: "ComfyUI Web",
            client_id: api.clientId,
            workflow: prompt["output"]
        }, null, 2)
    })
}


// Image background for nodes
// - this is just for visuals

function loadImage(base64) {
    const image = new Image()
    image.src = base64
    // image.onerror = () => console.error("Failed to load image");
    return image
}

const canvasIcon = loadImage("data:image/webp;base64,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")
const outputIcon = loadImage("data:image/webp;base64,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")

function setIconImage(nodeType, image, size, padRows, padCols) {
    const onAdded = nodeType.prototype.onAdded
    nodeType.prototype.onAdded = function () {
        onAdded?.apply(this, arguments)
        this.size = size
    }

    const onDrawBackground = nodeType.prototype.onDrawBackground
    nodeType.prototype.onDrawBackground = function(ctx) {
        onDrawBackground?.apply(this, arguments)

        const pad = [padCols * 20, LiteGraph.NODE_SLOT_HEIGHT * padRows + 8];
        if(this.flags.collapsed || pad[1] + 32 > this.size[1] || image.width === 0) {
            return
        }
        const avail = [this.size[0] - pad[0], this.size[1] - pad[1]]
        const scale = Math.min(1.0, avail[0] / image.width, avail[1] / image.height)
        const size = [Math.floor(image.width * scale), Math.floor(image.height * scale)]
        const offset = [Math.max(0, (avail[0] - size[0]) / 2), Math.max(0, (avail[1] - size[1]) / 2)]
        ctx.drawImage(image, offset[0], pad[1] + offset[1], size[0], size[1])
    }
}


// Parameter node
// - represents a customizable parameter that should be exposed in external tools
// - adapts to whichever node it is connected to, similar to the built-in "Primitive" node
// - can only be connected to slots which are converted widgets

const replaceableWidgets = ["INT", "FLOAT", "BOOLEAN", "STRING", "COMBO", "INT:seed"]

const parameterTypes = {
    "combo": ["choice"],
    "number": ["number", "number (integer)"],
    "toggle": ["toggle"],
    "text": ["text", "prompt (positive)", "prompt (negative)"],
}

function defaultParameterType(widgetType, connectedNode, connectedWidget) {
    let paramType = parameterTypes[widgetType][0]
    if (connectedNode.comfyClass === "CLIPTextEncode") {
        paramType = "prompt (positive)"
    }
    if (connectedWidget.options?.round === 1) {
        paramType = "number (integer)"
    }
    return paramType
}

function valueMatchesType(value, type, options) {
    if (type === "number") {
        return typeof value === "number"
    } else if (type === "combo") {
        return options?.values?.includes(value)
    } else if (type === "toggle") {
        return typeof value === "boolean"
    }
    return typeof value === "string"
}

function optionalWidgetValue(widgets, index, fallback) {
    const result = widgets.length > index ? widgets[index].value : null
    return result === null || result === -1e10 || result === 1e10 ? fallback : result
}

function changeWidgets(node, type, connectedNode, connectedWidget) {
    if (type === "customtext") {
        type = "text"
    }
    const options = connectedWidget.options

    const parameterTypeHint = node.widgets[1].value
    const notSpecialized = node.widgets[1].options.values.includes("auto")
    const parameterTypeMismatch = !parameterTypes[type].includes(parameterTypeHint)
    if (notSpecialized || parameterTypeMismatch) {
        node.widgets[1].options = {values: parameterTypes[type]}
    }
    if (parameterTypeMismatch) {
        node.widgets[1].value = defaultParameterType(type, connectedNode, connectedWidget)
    }
    const oldDefault = node.widgets.length > 2 ? node.widgets[2].value : connectedWidget.value
    const oldMin = optionalWidgetValue(node.widgets, 3, options?.min ?? 0)
    const oldMax = optionalWidgetValue(node.widgets, 4, options?.max ?? 100)
    const isDefaultValid = valueMatchesType(oldDefault, type, connectedWidget.options)
    while (node.widgets.length > 2) {
        node.widgets.pop()
    }
    const value = isDefaultValid && oldDefault !== "" ? oldDefault : connectedWidget.value
    node.addWidget(type, "default", value, null, options)
    if (type === "number") {
        node.addWidget("number", "min", oldMin, null, options)
        node.addWidget("number", "max", oldMax, null, options)
    }
}

function adaptWidgetsToConnection(node) {
    if (!node.outputs || node.outputs.length === 0) {
        return
    }
    const links = node.outputs[0].links
    if (links && links.length === 1) {
        const link = node.graph.links[links[0]]
        if (!link) return

        const theirNode = node.graph.getNodeById(link.target_id)
        if (!theirNode || !theirNode.inputs) return
    
        const input = theirNode.inputs[link.target_slot]
        if (!input || !input.widget || theirNode.widgets === undefined) return

        node.outputs[0].type = input.type

        if (node.widgets[0].value === "Parameter") {
            node.widgets[0].value = input.name
        }

        const widgetName = input.widget.name
        const theirWidget = theirNode.widgets.find((w) => w.name === widgetName)
        if (!theirWidget) return // connected to a custom node that isn't installed

        const widgetType = theirWidget.origType ?? theirWidget.type
        changeWidgets(node, widgetType, theirNode, theirWidget)

    } else if (!links || links.length === 0) {
        node.outputs[0].type = "*"
        node.widgets[1].value = "auto"
        node.widgets[1].options = {values: ["auto"]}
    }
}

function setupParameterNode(nodeType) {
    const onAdded = nodeType.prototype.onAdded
    nodeType.prototype.onAdded = function() {
        onAdded?.apply(this, arguments)
        adaptWidgetsToConnection(this)
    }

    const onAfterGraphConfigured = nodeType.prototype.onAfterGraphConfigured
    nodeType.prototype.onAfterGraphConfigured = function() {
        onAfterGraphConfigured?.apply(this, arguments)
        adaptWidgetsToConnection(this)
    }

    const onConnectOutput = nodeType.prototype.onConnectOutput
    nodeType.prototype.onConnectOutput = function(slot, type, input, target_node, target_slot) {
        if (!input.widget && !(input.type in replaceableWidgets)) {
            return false
        } else if (onConnectOutput) {
            result = onConnectOutput.apply(this, arguments)
            return result
        }
        return true
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = function(_, index, connected) {
        if (!app.configuringGraph) {
            adaptWidgetsToConnection(this)
        }
        onConnectionsChange?.apply(this, arguments)
    }
}


// Register the extension

app.registerExtension({
	name: "external_tooling_nodes",

    beforeRegisterNodeDef(nodeType /*typeof LGraphNode*/, nodeData /*ComfyObjectInfo*/, app) {
        if (nodeData.name === "ETN_KritaCanvas") {
            setIconImage(nodeType, canvasIcon, [200, 100], 0, 2)
        } else if (nodeData.name === "ETN_KritaOutput") {
            setIconImage(nodeType, outputIcon, [200, 100], 1, 0)
        } else if (nodeData.name === "ETN_Parameter") {
            setupParameterNode(nodeType)
        } else if (nodeData.name === "ETN_SendText") {
            const onAdded = nodeType.prototype.onAdded
            nodeType.prototype.onAdded = function() {
                onAdded?.apply(this, arguments)
                this.inputs[0].type = "*"
            }
        }
    },

    nodeCreated(node /*ComfyNode*/, app) {
        if (publisherRegistered || node.comfyClass !== "ETN_KritaOutput") {
            return
        }
        api.addEventListener('graphChanged', publishWorkflow)
        publisherRegistered = true
    },

    setup(app) {
        if (publisherRegistered) {
            publishWorkflow(null)
        }
    },
});

})();