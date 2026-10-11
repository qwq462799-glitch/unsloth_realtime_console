"""Web configuration API tests (run with: python -m unittest discover -s tests)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import auto_trainer_v3 as appmod


class ConfigApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="unsloth-console-tests-")
        cls.root = Path(cls.tmp.name)
        cls.original_paths = {
            "SYSTEM_SETTINGS_FILE": appmod.SYSTEM_SETTINGS_FILE,
            "SOURCE_CONFIG_FILE": appmod.SOURCE_CONFIG_FILE,
            "SOURCE_STATE_FILE": appmod.SOURCE_STATE_FILE,
        }
        appmod.SYSTEM_SETTINGS_FILE = cls.root / "system_settings.json"
        appmod.SOURCE_CONFIG_FILE = cls.root / "data_sources.json"
        appmod.SOURCE_STATE_FILE = cls.root / "source_state.json"
        cls.original_settings = copy.deepcopy(appmod.SYSTEM_SETTINGS)
        cls.original_sources = copy.deepcopy(appmod.DATA_SOURCES)
        cls.original_globals = {
            "MODEL_NAME": appmod.MODEL_NAME,
            "MAX_SEQ_LENGTH": appmod.MAX_SEQ_LENGTH,
            "BUFFER_THRESHOLD": appmod.BUFFER_THRESHOLD,
            "MAX_ADAPTERS": appmod.MAX_ADAPTERS,
            "DB_POLL_INTERVAL": appmod.DB_POLL_INTERVAL,
            "MAX_STEPS": appmod.MAX_STEPS,
            "TRAINING_BACKEND": appmod.TRAINING_BACKEND,
        }
        appmod.state.is_training = False
        cls.client = appmod.app.test_client()
        cls.headers = {"X-Auth-Token": appmod.AUTH_TOKEN} if appmod.AUTH_TOKEN else {}

    @classmethod
    def tearDownClass(cls):
        appmod.SYSTEM_SETTINGS.clear()
        appmod.SYSTEM_SETTINGS.update(cls.original_settings)
        appmod.DATA_SOURCES[:] = cls.original_sources
        for key, value in cls.original_globals.items():
            setattr(appmod, key, value)
        for key, value in cls.original_paths.items():
            setattr(appmod, key, value)
        cls.tmp.cleanup()

    def setUp(self):
        appmod.SYSTEM_SETTINGS.clear()
        appmod.SYSTEM_SETTINGS.update(copy.deepcopy(self.original_settings))
        appmod.DATA_SOURCES[:] = copy.deepcopy(self.original_sources)
        appmod.state.is_training = False
        for path_name in ("SYSTEM_SETTINGS_FILE", "SOURCE_CONFIG_FILE", "SOURCE_STATE_FILE"):
            Path(getattr(appmod, path_name)).unlink(missing_ok=True)

    def test_source_edit_and_secrets_are_redacted(self):
        response = self.client.get("/api/sources/1/config", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["fields"][4]["value"], "")
        response = self.client.put("/api/sources/1/config", headers=self.headers, json={"fields": {
            "name": "MySQL test", "host": "127.0.0.2", "port": "3307", "user": "trainer",
            "password": "example-secret", "database": "dataset", "poll_interval": "120",
            "sql": "SELECT id, instruction, output FROM qa_pairs WHERE id > %s ORDER BY id ASC LIMIT 200",
        }})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertTrue(Path(appmod.SOURCE_CONFIG_FILE).exists())
        reread = self.client.get("/api/sources/1/config", headers=self.headers).get_json()
        self.assertEqual(reread["fields"][4]["value"], "")
        self.assertTrue(reread["fields"][4]["configured"])

    def test_source_toggle_is_persistent(self):
        response = self.client.post("/api/sources/1/toggle", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        saved = json.loads(Path(appmod.SOURCE_STATE_FILE).read_text(encoding="utf-8"))
        self.assertEqual(saved["mysql"], response.json["enabled"])

    def test_system_config_can_be_disabled_and_falls_back(self):
        response = self.client.put("/api/config", headers=self.headers, json={"settings": {
            "max_steps": {"value": "9", "enabled": True},
        }})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(appmod.MAX_STEPS, 9)
        response = self.client.put("/api/config", headers=self.headers, json={"settings": {
            "max_steps": {"value": "11", "enabled": False},
        }})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(appmod.MAX_STEPS, appmod.DEFAULT_SYSTEM_VALUES["max_steps"])
        self.assertEqual(appmod.SYSTEM_SETTINGS["max_steps"]["value"], 11)

    def test_invalid_config_is_rejected(self):
        response = self.client.put("/api/config", headers=self.headers, json={"settings": {
            "db_poll_interval": {"value": "1", "enabled": True},
        }})
        self.assertEqual(response.status_code, 400)

    def test_sql_source_rejects_mutating_statements(self):
        response = self.client.put("/api/sources/1/config", headers=self.headers, json={"fields": {
            "sql": "SELECT id, instruction, output FROM qa_pairs WHERE id > %s ORDER BY id ASC LIMIT 200; DROP TABLE qa_pairs"
        }})
        self.assertEqual(response.status_code, 400)

    def test_http_url_masks_and_preserves_query_credentials(self):
        idx = next(i for i, source in enumerate(appmod.DATA_SOURCES) if source["type"] == "http_api")
        original_url = "https://user:topsecret@example.com/qa?api_key=api-secret&cursor=2"
        appmod.DATA_SOURCES[idx]["url"] = original_url
        response = self.client.get(f"/api/sources/{idx}/config", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        fields = {field["key"]: field for field in response.json["fields"]}
        self.assertNotIn("topsecret", fields["url"]["value"])
        self.assertNotIn("api-secret", fields["url"]["value"])
        response = self.client.put(f"/api/sources/{idx}/config", headers=self.headers, json={"fields": {
            "url": fields["url"]["value"]
        }})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(appmod.DATA_SOURCES[idx]["url"], original_url)
        changed_host_url = fields["url"]["value"].replace("example.com", "other.example")
        response = self.client.put(f"/api/sources/{idx}/config", headers=self.headers, json={"fields": {
            "url": changed_host_url
        }})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(appmod.DATA_SOURCES[idx]["url"], original_url)


if __name__ == "__main__":
    unittest.main()
