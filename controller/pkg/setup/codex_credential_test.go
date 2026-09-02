package setup

import (
	"context"
	"testing"

	api "github.com/agentgateway/agentgateway/api"
	"github.com/agentgateway/agentgateway/controller/pkg/syncer/krtxds"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"istio.io/istio/pkg/security"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func credentialContext(namespace string) context.Context {
	return context.WithValue(context.Background(), krtxds.PeerCtxKey, &security.Caller{
		KubernetesInfo: security.KubernetesInfo{PodNamespace: namespace},
	})
}

func TestCodexCredentialStoreEncryptsNamespaceLocalRecordAndUsesCAS(t *testing.T) {
	ctx := context.Background()
	kube := fake.NewSimpleClientset()
	store, err := newCodexCredentialStore(ctx, kube, "controller-system")
	if err != nil {
		t.Fatal(err)
	}
	callerCtx := credentialContext("tenant-a")
	created, err := store.Replace(callerCtx, &api.CodexCredentialReplaceRequest{Credential: []byte("credential-record")})
	if err != nil {
		t.Fatal(err)
	}
	secret, err := kube.CoreV1().Secrets("tenant-a").Get(ctx, codexCredentialSecretName, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if string(secret.Data[codexCredentialDataKey]) == "credential-record" {
		t.Fatal("credential was stored in plaintext")
	}
	loaded, err := store.Load(callerCtx, &api.CodexCredentialLoadRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if string(loaded.Credential) != "credential-record" || loaded.Generation != created.Generation {
		t.Fatalf("unexpected loaded credential metadata")
	}
	_, err = store.Replace(callerCtx, &api.CodexCredentialReplaceRequest{ExpectedGeneration: "stale", Credential: []byte("new-record")})
	if status.Code(err) != codes.Aborted {
		t.Fatalf("stale replace code = %v, want %v", status.Code(err), codes.Aborted)
	}
}

func TestCodexCredentialStoreRejectsTamperedRecordAndUnauthenticatedCaller(t *testing.T) {
	ctx := context.Background()
	kube := fake.NewSimpleClientset(&corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: codexCredentialKeySecret, Namespace: "controller-system"},
		Data:       map[string][]byte{codexCredentialKeyDataKey: make([]byte, 32)},
	})
	store, err := newCodexCredentialStore(ctx, kube, "controller-system")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load(ctx, &api.CodexCredentialLoadRequest{}); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("unauthenticated load code = %v, want %v", status.Code(err), codes.Unauthenticated)
	}
	_, err = kube.CoreV1().Secrets("tenant-a").Create(ctx, &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: codexCredentialSecretName},
		Data:       map[string][]byte{codexCredentialDataKey: []byte("tampered")},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load(credentialContext("tenant-a"), &api.CodexCredentialLoadRequest{}); status.Code(err) != codes.Internal {
		t.Fatalf("tampered load code = %v, want %v", status.Code(err), codes.Internal)
	}
}
