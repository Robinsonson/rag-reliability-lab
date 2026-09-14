import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from document_library import DocumentLibrary
from lab_api import app


def encoded(text):
    return base64.b64encode(text.encode()).decode()


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.library = DocumentLibrary(self.temp.name)
        self.patch = patch('lab_api._library', return_value=self.library)
        self.patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.patch.stop()
        self.library.close()
        self.temp.cleanup()

    def test_replacement_is_atomic_and_original_version_remains_readable(self):
        first = self.library.upload('policy.txt', encoded('Transport ceiling: SGD 80.'))
        with self.assertRaises(ValueError):
            self.library.upload('bad.txt', encoded('   '), first['id'])
        self.assertTrue(self.library.catalog()['documents'][0]['active'])
        second = self.library.upload('policy.txt', encoded('Transport ceiling: SGD 40.'), first['id'])
        rows = self.library.catalog()['documents']
        self.assertEqual([r['id'] for r in rows if r['active']], [second['id']])
        self.assertEqual(second['version'], 2)
        self.assertIn(b'80', self.library.original(first['id'])[1])
        self.assertIn('80', self.client.get(f"/api/documents/{first['id']}/pages/1").text)
        with self.assertRaises(ValueError):
            self.library.upload('policy.txt', encoded('Third revision'), first['id'])
        self.library.activate(first['id'], True)
        self.assertEqual([r['id'] for r in self.library.catalog()['documents'] if r['active']], [first['id']])

    def test_api_rejects_bad_files_and_ambiguous_library_page_labels(self):
        for name, data in [('x.html', encoded('<script>bad</script>')), ('x.pdf', encoded('not PDF')), ('x.txt', 'invalid!')]:
            response = self.client.post('/api/documents', json={'name':name, 'data':data})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.library.catalog()['documents'], [])
        self.assertEqual(self.client.post('/api/retrieval/compare', json={'query':'policy', 'corpus':'library','expected_pages':[1]}).status_code, 400)
        self.assertEqual(self.client.get('/api/documents/missing/original').status_code, 404)

    def test_metadata_and_activation_persist_across_restart(self):
        doc = self.library.upload('../../policy.txt', encoded('Test policy'))
        self.library.activate(doc['id'], False)
        self.library.close()
        self.library = DocumentLibrary(self.temp.name)
        row = self.library.catalog()['documents'][0]
        self.assertEqual(row['name'], 'policy.txt')
        self.assertFalse(row['active'])
        self.assertEqual(self.library.catalog()['index_status'], 'empty')

    def test_failed_build_does_not_reuse_old_runtime(self):
        self.library.upload('one.txt', encoded('One policy'))
        with patch('rag_engine.get_embeddings', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):
                self.library.runtime()
        self.assertEqual(self.library.catalog()['index_status'], 'failed')
        self.assertIsNone(self.library._runtime)

    def test_library_request_never_uses_demo_runtime(self):
        documents = []
        dense = Mock(); dense.invoke.return_value = documents
        advanced = Mock(); advanced.base_retriever.invoke.return_value = documents
        advanced.base_compressor.compress_documents.return_value = documents
        with patch.object(self.library, 'runtime', return_value=(None, [], dense, advanced)), patch('lab_api._cached_runtime') as demo:
            result = self.client.post('/api/retrieval/compare', json={'query':'test question','corpus':'library'})
            self.assertEqual(result.status_code, 200)
            demo.assert_not_called()

    def test_text_pdf_import_and_exact_original_download(self):
        source = Path(__file__).resolve().parents[1] / 'company_policy.pdf'
        raw = source.read_bytes()
        result = self.client.post('/api/documents', json={'name':'policy.pdf','data':base64.b64encode(raw).decode()})
        self.assertEqual(result.status_code, 201)
        identifier = result.json()['id']
        self.assertEqual(self.client.get(f'/api/documents/{identifier}/original').content, raw)
        self.assertEqual(self.library.catalog()['documents'][0]['page_count'],20)
        self.assertEqual(self.client.get(f'/api/documents/{identifier}/pages/21').status_code,404)
