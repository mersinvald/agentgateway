package setup

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"fmt"
	"io"

	api "github.com/agentgateway/agentgateway/api"
	"github.com/agentgateway/agentgateway/controller/pkg/syncer/krtxds"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"istio.io/istio/pkg/security"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const (
	codexCredentialSecretName = "agentgateway-codex-oauth"
	codexCredentialDataKey    = "credential"
	codexCredentialKeySecret  = "agentgateway-codex-credential-key"
	codexCredentialKeyDataKey = "key"
	maxCodexCredentialSize    = 64 * 1024
)

// codexCredentialStore encrypts OAuth records before Kubernetes persistence.
type codexCredentialStore struct {
	api.UnimplementedCodexCredentialServiceServer
	kube kubernetes.Interface
	key  []byte
}

func newCodexCredentialStore(ctx context.Context, kubeClient kubernetes.Interface, controllerNamespace string) (*codexCredentialStore, error) {
	secrets := kubeClient.CoreV1().Secrets(controllerNamespace)
	secret, err := secrets.Get(ctx, codexCredentialKeySecret, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		key := make([]byte, 32)
		if _, err := io.ReadFull(rand.Reader, key); err != nil {
			return nil, err
		}
		secret, err = secrets.Create(ctx, &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: codexCredentialKeySecret}, Type: corev1.SecretTypeOpaque, Data: map[string][]byte{codexCredentialKeyDataKey: key}}, metav1.CreateOptions{})
		if apierrors.IsAlreadyExists(err) {
			secret, err = secrets.Get(ctx, codexCredentialKeySecret, metav1.GetOptions{})
		}
	}
	if err != nil {
		return nil, fmt.Errorf("load Codex credential encryption key: %w", err)
	}
	key := secret.Data[codexCredentialKeyDataKey]
	if len(key) != 32 {
		return nil, fmt.Errorf("Codex credential encryption key is invalid")
	}
	return &codexCredentialStore{kube: kubeClient, key: key}, nil
}

func (s *codexCredentialStore) Load(ctx context.Context, _ *api.CodexCredentialLoadRequest) (*api.CodexCredentialLoadResponse, error) {
	namespace, err := callerNamespace(ctx)
	if err != nil {
		return nil, err
	}
	secret, err := s.kube.CoreV1().Secrets(namespace).Get(ctx, codexCredentialSecretName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return &api.CodexCredentialLoadResponse{}, nil
	}
	if err != nil {
		return nil, status.Error(codes.Internal, "credential storage unavailable")
	}
	credential, err := s.decrypt(secret.Data[codexCredentialDataKey])
	if err != nil {
		return nil, status.Error(codes.Internal, "credential storage unavailable")
	}
	return &api.CodexCredentialLoadResponse{Credential: credential, Generation: secret.ResourceVersion}, nil
}

func (s *codexCredentialStore) Replace(ctx context.Context, req *api.CodexCredentialReplaceRequest) (*api.CodexCredentialReplaceResponse, error) {
	if len(req.GetCredential()) == 0 || len(req.GetCredential()) > maxCodexCredentialSize {
		return nil, status.Error(codes.InvalidArgument, "invalid credential record")
	}
	namespace, err := callerNamespace(ctx)
	if err != nil {
		return nil, err
	}
	encrypted, err := s.encrypt(req.GetCredential())
	if err != nil {
		return nil, status.Error(codes.Internal, "credential storage unavailable")
	}
	secrets := s.kube.CoreV1().Secrets(namespace)
	secret, err := secrets.Get(ctx, codexCredentialSecretName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		if req.GetExpectedGeneration() != "" {
			return nil, status.Error(codes.Aborted, "credential generation changed")
		}
		created, createErr := secrets.Create(ctx, &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: codexCredentialSecretName}, Type: corev1.SecretTypeOpaque, Data: map[string][]byte{codexCredentialDataKey: encrypted}}, metav1.CreateOptions{})
		if apierrors.IsAlreadyExists(createErr) {
			return nil, status.Error(codes.Aborted, "credential generation changed")
		}
		if createErr != nil {
			return nil, status.Error(codes.Internal, "credential storage unavailable")
		}
		return &api.CodexCredentialReplaceResponse{Generation: created.ResourceVersion}, nil
	}
	if err != nil {
		return nil, status.Error(codes.Internal, "credential storage unavailable")
	}
	if secret.ResourceVersion != req.GetExpectedGeneration() {
		return nil, status.Error(codes.Aborted, "credential generation changed")
	}
	secret.Data = map[string][]byte{codexCredentialDataKey: encrypted}
	updated, err := secrets.Update(ctx, secret, metav1.UpdateOptions{})
	if apierrors.IsConflict(err) {
		return nil, status.Error(codes.Aborted, "credential generation changed")
	}
	if err != nil {
		return nil, status.Error(codes.Internal, "credential storage unavailable")
	}
	return &api.CodexCredentialReplaceResponse{Generation: updated.ResourceVersion}, nil
}

func callerNamespace(ctx context.Context) (string, error) {
	caller, ok := ctx.Value(krtxds.PeerCtxKey).(*security.Caller)
	if !ok || caller == nil || caller.KubernetesInfo.PodNamespace == "" {
		return "", status.Error(codes.Unauthenticated, "credential API requires an authenticated Kubernetes workload")
	}
	return caller.KubernetesInfo.PodNamespace, nil
}

func (s *codexCredentialStore) encrypt(plaintext []byte) ([]byte, error) {
	block, err := aes.NewCipher(s.key)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}
	return gcm.Seal(nonce, nonce, plaintext, nil), nil
}

func (s *codexCredentialStore) decrypt(encrypted []byte) ([]byte, error) {
	block, err := aes.NewCipher(s.key)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil || len(encrypted) < gcm.NonceSize() {
		return nil, fmt.Errorf("invalid encrypted credential")
	}
	return gcm.Open(nil, encrypted[:gcm.NonceSize()], encrypted[gcm.NonceSize():], nil)
}
