import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
	name: "external_tooling_nodes",
    init() {
        api.addEventListener('graphChanged', async (e) => {
            const prompt = await app.graphToPrompt();
            console.log('graphChanged', prompt);
            api.fetchApi("/api/etn/workflow/publish", {
                method: "POST",
                body: JSON.stringify({
                    name: "ComfyUI Web",
                    client_id: api.clientId,
                    workflow: prompt["output"]
                }, null, 2)
            });
        });
    }
});
