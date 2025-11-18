import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Upload, Database, FileText, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { Dataset } from "@shared/schema";

export default function Datasets() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState("");
  const [previewData, setPreviewData] = useState<any[] | null>(null);
  const { toast } = useToast();

  const { data: datasets } = useQuery<Dataset[]>({
    queryKey: ["/api/datasets"],
  });

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await fetch("/api/datasets/upload", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Upload failed");
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/datasets"] });
      toast({
        title: "Dataset uploaded successfully",
        description: "Your dataset is now available for training",
      });
      setSelectedFile(null);
      setDatasetName("");
      setPreviewData(null);
    },
    onError: () => {
      toast({
        title: "Upload failed",
        description: "There was an error uploading your dataset",
        variant: "destructive",
      });
    },
  });

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setDatasetName(file.name.replace(/\.[^/.]+$/, ""));

      // Preview CSV
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        const lines = text.split("\n").slice(0, 6); // Header + 5 rows
        const parsed = lines.map(line =>
          line.split(",").map(cell => cell.trim())
        );
        setPreviewData(parsed);
      };
      reader.readAsText(file);
    }
  };

  const handleUpload = () => {
    if (!selectedFile || !datasetName) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("name", datasetName);

    uploadMutation.mutate(formData);
  };

  return (
    <div className="space-y-8">
      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-xl font-semibold">Upload Dataset</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Upload Zone */}
            <div className="space-y-4">
              <div
                className="border-2 border-dashed border-border rounded-md p-8 text-center hover-elevate transition-colors"
                onDragOver={(e) => {
                  e.preventDefault();
                  e.currentTarget.classList.add("border-primary", "bg-primary/5");
                }}
                onDragLeave={(e) => {
                  e.currentTarget.classList.remove("border-primary", "bg-primary/5");
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.currentTarget.classList.remove("border-primary", "bg-primary/5");
                  const file = e.dataTransfer.files[0];
                  if (file && file.name.endsWith('.csv')) {
                    setSelectedFile(file);
                    setDatasetName(file.name.replace(/\.[^/.]+$/, ""));
                    const reader = new FileReader();
                    reader.onload = (event) => {
                      const text = event.target?.result as string;
                      const lines = text.split("\n").slice(0, 6);
                      const parsed = lines.map(line =>
                        line.split(",").map(cell => cell.trim())
                      );
                      setPreviewData(parsed);
                    };
                    reader.readAsText(file);
                  }
                }}
              >
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                  data-testid="input-file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer flex flex-col items-center gap-2"
                >
                  <Upload className="w-12 h-12 text-muted-foreground" />
                  <div className="text-sm font-medium text-foreground">
                    {selectedFile ? selectedFile.name : "Drag & drop CSV or click to upload"}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    CSV files only • Max 100MB
                  </div>
                </label>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Dataset Name</label>
                <Input
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  placeholder="Enter dataset name"
                  data-testid="input-dataset-name"
                />
              </div>

              <Button
                onClick={handleUpload}
                disabled={!selectedFile || !datasetName || uploadMutation.isPending}
                className="w-full"
                data-testid="button-upload-dataset"
              >
                {uploadMutation.isPending ? "Uploading..." : "Upload Dataset"}
              </Button>
            </div>

            {/* Preview */}
            <div>
              <h3 className="text-sm font-medium text-foreground mb-3">Preview</h3>
              {previewData ? (
                <div className="border border-border rounded-md overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs font-mono">
                      <thead>
                        <tr className="bg-muted">
                          {previewData[0]?.map((header, idx) => (
                            <th
                              key={idx}
                              className="px-3 py-2 text-left font-semibold text-foreground border-b border-border"
                            >
                              {header}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {previewData.slice(1).map((row, rowIdx) => (
                          <tr key={rowIdx} className="border-b border-border last:border-0">
                            {row.map((cell, cellIdx) => (
                              <td key={cellIdx} className="px-3 py-2 text-muted-foreground">
                                {cell}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="px-3 py-2 bg-muted text-xs text-muted-foreground">
                    Showing first 5 rows
                  </div>
                </div>
              ) : (
                <div className="border border-border rounded-md h-[250px] flex items-center justify-center text-sm text-muted-foreground">
                  Upload a file to preview
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Dataset List */}
      <Card>
        <CardHeader>
          <CardTitle className="text-xl font-semibold">Available Datasets</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {datasets?.map((dataset) => (
              <div
                key={dataset.id}
                className="flex items-center justify-between gap-4 p-4 rounded-md border border-border hover-elevate"
                data-testid={`dataset-${dataset.id}`}
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center">
                    <Database className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-foreground">{dataset.name}</div>
                    <div className="text-xs text-muted-foreground font-mono">{dataset.filename}</div>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <div className="text-xs text-muted-foreground">
                      {dataset.rowCount.toLocaleString()} rows
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {dataset.columnCount} columns
                    </div>
                  </div>
                  <Badge variant="secondary" className="font-mono text-xs">
                    {new Date(dataset.uploadedAt).toLocaleDateString()}
                  </Badge>
                </div>
              </div>
            ))}
            {(!datasets || datasets.length === 0) && (
              <div className="text-center py-12 text-muted-foreground">
                <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <div className="text-sm font-medium">No datasets uploaded yet</div>
                <div className="text-xs mt-1">Upload your first dataset to begin</div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
