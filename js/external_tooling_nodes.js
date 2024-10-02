import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

(function() {

let publisherRegistered = false;

async function publishWorkflow(e) {
    const prompt = await app.graphToPrompt();
    await api.fetchApi("/api/etn/workflow/publish", {
        method: "POST",
        body: JSON.stringify({
            name: "ComfyUI Web",
            client_id: api.clientId,
            workflow: prompt["output"]
        }, null, 2)
    });
}

app.registerExtension({
	name: "external_tooling_nodes",

    nodeCreated(node /*: ComfyNode */, app /*: ComfyApp */) {
        if (publisherRegistered || node.comfyClass !== "ETN_KritaOutput") {
            return;
        }
        console.log("nodeCreated");
        api.addEventListener('graphChanged', publishWorkflow);
        publisherRegistered = true;
    },

    setup(app) {
        if (publisherRegistered) {
            publishWorkflow(null);
        }
    },

});

})();